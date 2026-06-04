import os
from collections import defaultdict
import imageio
import torch
import numpy as np
import wandb
import glob


def concat_batches(offline_batch, online_batch, axis=1):
    batch = defaultdict(list)

    if isinstance(offline_batch, dict) and isinstance(online_batch, dict):
        for k, v in offline_batch.items():
            if isinstance(v, dict):
                batch[k] = concat_batches(offline_batch[k], online_batch[k], axis=axis)
            else:
                if isinstance(v, torch.Tensor) and isinstance(online_batch[k], torch.Tensor):
                    batch[k] = torch.cat((v, online_batch[k]), dim=axis)
                elif isinstance(v, np.ndarray) and isinstance(online_batch[k], np.ndarray):
                    batch[k] = np.concatenate((v, online_batch[k]), axis=axis)
                else:
                    raise TypeError(f"Unsupported type for concatenation: {type(v)} and {type(online_batch[k])}")
    return batch

def load_recorded_video(video_path: str):
    video = np.array(imageio.mimread(video_path, "MP4")).transpose((0, 3, 1, 2))
    assert video.shape[1] == 3, "Numpy array should be (T, C, H, W)"
    return wandb.Video(video, fps=20)

def state_dict_to_numpy(state_dict, skip_optimizer=True):
    result = {}
    for k, v in state_dict.items():
        if skip_optimizer and "optimizer" in k:
            continue
        if k == "config":
            continue
            
        if isinstance(v, torch.Tensor):
            result[k] = v.detach().cpu().numpy()
        elif isinstance(v, dict):
            result[k] = state_dict_to_numpy(v, skip_optimizer=False)
        else:
            continue
    return result

def numpy_to_state_dict(params, device):
    result = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.as_tensor(v, device=device)
        elif isinstance(v, dict):
            result[k] = numpy_to_state_dict(v, device)
        else:
            result[k] = v
    return result

def print_green(x):
    return print("\033[92m {}\033[00m".format(x))

def save_checkpoint(model, optimizer, global_step, checkpoint_path):
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    
    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_step_{global_step}.pt")
    torch.save(checkpoint, checkpoint_file)
    
    latest_file = os.path.join(checkpoint_path, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_file)
    
    print_green(f"Saved checkpoint to {checkpoint_file}")


def load_checkpoint(model, optimizer=None, checkpoint_path=None, device="cuda"):
    if os.path.isdir(checkpoint_path):
        latest_file = os.path.join(checkpoint_path, "checkpoint_latest.pt")
        if os.path.exists(latest_file):
            checkpoint_file = latest_file
        else:
            checkpoint_files = glob.glob(os.path.join(checkpoint_path, "checkpoint_*.pt"))
            if not checkpoint_files:
                return
            checkpoint_file = max(checkpoint_files, key=os.path.getctime)
    else:
        checkpoint_file = checkpoint_path
    
    checkpoint = torch.load(checkpoint_file, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print_green(f"Loaded checkpoint from {checkpoint_file}")
    return model, optimizer