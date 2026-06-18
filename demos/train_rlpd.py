import warnings
warnings.filterwarnings("ignore")

import logging
import glob
logging.getLogger('asyncio').setLevel(logging.ERROR)

import time
import numpy as np
import tqdm
from absl import app, flags
import os
import copy

from typing import Optional
import pickle as pkl
from gymnasium.wrappers import RecordEpisodeStatistics

import torch
from launcher.utils.torch_utils import dict_apply
from launcher.utils.timer_utils import Timer
from launcher.utils.train_utils import (
    concat_batches, 
    state_dict_to_numpy, 
    numpy_to_state_dict, 
    print_green
)

from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from launcher.data.data_store import ReplayBufferDataStore
from launcher import (
    make_trainer_config, 
    make_wandb_logger, 

    make_sac_pixel_agent,
    make_sac_pointcloud_agent,
)

from demos.experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")

flags.DEFINE_boolean(
    "debug", False, "Debug mode."
) 

flags.DEFINE_boolean("use_amp", True, "Use mixed precision training (AMP) for faster training.")

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
flags.DEFINE_string("agent_type", "sac", "Agent type.")


def actor(
    agent, 
    data_store, 
    intvn_data_store, 
    env,
    device: str = "cuda"
):
    agent.eval()
    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }
    
    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
    )

    def update_params(params):
        """Update agent parameters from server"""
        state_dict = numpy_to_state_dict(params, device)
        agent.load_state_dict(state_dict, strict=False)

    client.recv_network_callback(update_params)

    transitions = []
    demo_transitions = []

    obs, _ = env.reset()
    done = False

    timer = Timer()
    running_return = 0.0
    intervention_count = 0
    intervention_steps = 0
    already_intervened = False

    pbar = tqdm.tqdm(range(config.max_steps), dynamic_ncols=True)
    for step in pbar:
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < config.random_steps:
                actions = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = dict_apply(obs, lambda x: torch.as_tensor(x, device=device))
                    actions = agent.sample_actions(
                        observations=obs_tensor,
                        argmax=False,
                    )
                actions = actions.cpu().numpy()

        # Step environment
        with timer.context("step_env"):
            next_obs, reward, done, truncated, info = env.step(actions)
            reward = np.asarray(reward, dtype=np.float32)

            if "total_action" in info:
                actions = info.pop("total_action")
            
            # Track intervention statistics
            if "intervene_action" in info:
                actions = info.pop("intervene_action")
                intervention_steps += 1
                if not already_intervened:
                    intervention_count += 1
                already_intervened = True
            else:
                already_intervened = False
            
            running_return += reward

            transition = dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=reward,
                masks=1.0 - done,
                dones=done or truncated,
            )
            
            # All data goes into replay buffer
            data_store.insert(transition)
            transitions.append(copy.deepcopy(transition))
            
            # Intervention data additionally goes into intervention buffer
            if already_intervened:
                intvn_data_store.insert(transition)
                demo_transitions.append(copy.deepcopy(transition))

            obs = next_obs
            if done or truncated:
                # Add intervention statistics to episode info
                if "episode" in info:
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                
                stats = {"environment": info}
                client.request("send-stats", stats)
                pbar.set_description(f"Return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                already_intervened = False
                client.update()
                obs, _ = env.reset()

        if step > 0 and config.buffer_period > 0 and step % config.buffer_period == 0:
            buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer")
            if not os.path.exists(buffer_path):
                os.makedirs(buffer_path)
            with open(os.path.join(buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                pkl.dump(transitions, f)
                transitions = []

            if len(demo_transitions) > 0:
                demo_buffer_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer")
                if not os.path.exists(demo_buffer_path):
                    os.makedirs(demo_buffer_path)
                with open(os.path.join(demo_buffer_path, f"transitions_{step}.pkl"), "wb") as f:
                    pkl.dump(demo_transitions, f)
                    demo_transitions = []

        timer.tock("total")

        if step % config.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


def learner(
    agent,
    replay_buffer: ReplayBufferDataStore,
    demo_buffer: Optional[ReplayBufferDataStore] = None,
    device: str = "cuda",
):
    agent.train()
    wandb_logger = make_wandb_logger(
        project="serl-plus-plus",
        description=FLAGS.exp_name,
        debug=FLAGS.debug,
    )

    step = 0
    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=step)
        return {}

    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)

    pbar = tqdm.tqdm(
        total=config.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < config.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)
    pbar.close()

    server.publish_network(state_dict_to_numpy(agent.state_dict()))
    print_green("sent initial network to actor")

    if demo_buffer:
        single_buffer_batch_size = config.batch_size // 2
        demo_iterator = demo_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
        },
        device=device)
    else:
        single_buffer_batch_size = config.batch_size
        demo_iterator = None
    
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
        },
        device=device,
    )

    timer = Timer()

    pbar = tqdm.tqdm(total=config.replay_buffer_capacity,
                     initial=len(replay_buffer), desc="replay buffer")

    for step in tqdm.tqdm(range(config.max_steps), dynamic_ncols=True, desc="learner"):
        for _ in range(config.cta_ratio - 1):
            with timer.context("train_critics"):
                batch = next(replay_iterator)
                if demo_iterator:
                    demo_batch = next(demo_iterator)
                    batch = concat_batches(batch, demo_batch, axis=0)

                agent.update(batch, networks_to_update=frozenset({"critic"}))

        with timer.context("train"):
            batch = next(replay_iterator)
            if demo_iterator:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            update_info = agent.update(batch, 
                networks_to_update=frozenset({"actor", "critic", "temperature"})
            )

        if step > 0 and step % (config.steps_per_update) == 0:
            torch.cuda.synchronize()
            with torch.no_grad():
                state_dict = agent.state_dict()
                numpy_params = state_dict_to_numpy(state_dict)
            server.publish_network(numpy_params)
            del state_dict, numpy_params
            torch.cuda.empty_cache()

        if step % config.log_period == 0 and wandb_logger:
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if config.checkpoint_period and step > 0 and step % config.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            os.makedirs(FLAGS.checkpoint_path, exist_ok=True)
            checkpoint_file = os.path.join(FLAGS.checkpoint_path, f"checkpoint_{step}.pt")

            with torch.no_grad():
                torch.save({'step': step, 'model_state_dict': agent.state_dict()}, checkpoint_file)
            print_green(f"Saved checkpoint to {checkpoint_file}")
            torch.cuda.empty_cache()

        pbar.update(len(replay_buffer) - pbar.n)


def main(_):
    global config
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_green(f"Using device: {device}")
    
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(FLAGS.seed)

    assert FLAGS.exp_name in CONFIG_MAPPING, "Experiment folder not found."
    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=False,
        classifier=False,
    )
    env = RecordEpisodeStatistics(env)

    if FLAGS.agent_type == "sac":
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount
        )
    elif FLAGS.agent_type == "sac3":
        agent = make_sac_pointcloud_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount
        )
    else:
        raise NotImplementedError(f"Agent type {FLAGS.agent_type} not implemented.")
        
    agent = agent.to(device)

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        checkpoint_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*.pt"))
        if checkpoint_files:
            input("Checkpoint path already exists. Press Enter to resume training.")
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            ckpt = torch.load(latest_checkpoint, map_location=device)
            agent.load_state_dict(ckpt['model_state_dict'], strict=False)
            print_green(f"Loaded previous checkpoint at step {ckpt['step']} from {latest_checkpoint}.")
        else:
            print_green(f"Checkpoint directory exists but no checkpoint files found.")

    if FLAGS.learner:
        replay_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            device="cpu"
        )
        
        demo_buffer = ReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            device="cpu",
        )
        print_green("replay buffer created")

        if FLAGS.demo_path:
            for path in FLAGS.demo_path:
                with open(path, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        if 'infos' in transition and 'grasp_penalty' in transition['infos']:
                            transition['grasp_penalty'] = transition['infos']['grasp_penalty']
                        demo_buffer.insert(transition)
        else:
            print_green("No demo path provided. Creating empty demo buffer.")
            demo_buffer = None
        
        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        replay_buffer.insert(transition)
            print_green(
                f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}"
            )

        if FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ) and demo_buffer is not None:
            for file in glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl")):
                with open(file, "rb") as f:
                    transitions = pkl.load(f)
                    for transition in transitions:
                        demo_buffer.insert(transition)
            print_green(
                f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}"
            )

        print_green(f"replay_buffer size: {len(replay_buffer)}")
        print_green(f"demo buffer size: {len(demo_buffer) if demo_buffer else 0}")

        print_green("starting learner loop")
        learner(agent, replay_buffer=replay_buffer, demo_buffer=demo_buffer, device=device)

    elif FLAGS.actor:
        data_store = QueuedDataStore(50000)
        demo_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(agent, data_store, demo_data_store, env, device=device)

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    app.run(main)