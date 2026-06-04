import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import glob
import torch

from launcher.launch import make_sac_pixel_agent
from launcher.utils.torch_utils import dict_apply
from launcher.utils.data_utils import compute_mc_returns
from demos.experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("checkpoint_path", None, "Path to load checkpoint.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)

    if FLAGS.checkpoint_path is not None and os.path.exists(FLAGS.checkpoint_path):
        agent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        checkpoint_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "checkpoint_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            ckpt = torch.load(latest_checkpoint, map_location=device)
            agent.load_state_dict(ckpt['model_state_dict'], strict=False)
            print(f"Loaded previous checkpoint at step {ckpt['step']} from {latest_checkpoint}.")
        else:
            print(f"Checkpoint directory exists but no checkpoint files found.")

        agent = agent.to(device)
        agent.eval()
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        if FLAGS.checkpoint_path is not None:
            with torch.no_grad():
                obs_tensor = dict_apply(obs, lambda x: torch.as_tensor(x, device=device))
                actions = agent.sample_actions(observations=obs_tensor, argmax=True)
                actions = actions.cpu().numpy()

        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done or truncated:
            if info["succeed"]:
                trajectory = compute_mc_returns(trajectory, config.discount)
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)
