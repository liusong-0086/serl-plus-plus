import os
import pickle as pkl
from typing import List
import numpy as np
import zarr 
from numcodecs import Blosc
import gc
import hashlib


def convert_pickle_to_zarr(pkl_files: List[str],
    zarr_path: str, chunk_size: int = 1000):

    print("Pass 1/2: Scanning data structure...")
    total_count = 0
    obs_keys = set()
    sample_obs_shapes = {}
    sample_action_shape = None
    episode_ends_list = []  # Track episode boundaries
    
    for pkl_file in pkl_files:
        print(f"  Scanning: {os.path.basename(pkl_file)}")
        with open(pkl_file, "rb") as f:
            transitions = pkl.load(f)
        
        if not isinstance(transitions, list):
            transitions = [transitions]
        
        # Track episode boundaries using 'dones' field
        for i, trans in enumerate(transitions):
            if trans.get("dones", False):
                episode_ends_list.append(total_count + i + 1)
        
        total_count += len(transitions)
        
        # Get structure from first transition
        if transitions:
            trans = transitions[0]
            if "observations" in trans:
                # Flatten nested observation dict
                flat_obs = flatten_obs_dict(trans["observations"])
                for key, value in flat_obs.items():
                    obs_keys.add(key)
                    if key not in sample_obs_shapes:
                        arr = np.array(value)
                        # Squeeze out unnecessary leading dimensions
                        while arr.ndim > 1 and arr.shape[0] == 1:
                            arr = arr.squeeze(0)
                        sample_obs_shapes[key] = arr.shape
                        print(f"    Found obs key '{key}' with shape {arr.shape}")
            
            if "actions" in trans and sample_action_shape is None:
                arr = np.array(trans["actions"])
                sample_action_shape = arr.shape
                print(f"    Found action with shape {arr.shape}")
        
        del transitions
        gc.collect()
    
    # If no episode boundaries found, treat entire dataset as one episode
    if not episode_ends_list:
        episode_ends_list = [total_count]
    # Make sure the last transition is an episode end
    elif episode_ends_list[-1] != total_count:
        episode_ends_list.append(total_count)
    
    print(f"  Total transitions: {total_count}")
    print(f"  Number of episodes: {len(episode_ends_list)}")
    
    # Create zarr store with pre-allocated arrays
    print("Pass 2/2: Converting to zarr...")
    
    if os.path.exists(zarr_path):
        import shutil
        shutil.rmtree(zarr_path)
    
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open(zarr_path, mode="w")
    
    # Save episode ends
    root.create_dataset(
        "episode_ends",
        data=np.array(episode_ends_list, dtype=np.int64),
        compressor=compressor
    )
    
    # Create datasets
    action_shape = (total_count,) + sample_action_shape
    action_chunks = (min(chunk_size, total_count),) + sample_action_shape
    root.create_dataset(
        "actions",
        shape=action_shape,
        dtype=np.float32,
        chunks=action_chunks,
        compressor=compressor
    )
    
    obs_group = root.create_group("observations")
    for key in obs_keys:
        obs_shape = (total_count,) + sample_obs_shapes[key]
        # Use appropriate dtype based on shape (images are uint8, states are float32)
        if len(sample_obs_shapes[key]) == 3:  # Image (H, W, C)
            dtype = np.uint8
            chunks = (min(100, total_count),) + sample_obs_shapes[key]
        else:
            dtype = np.float32
            chunks = (min(chunk_size, total_count),) + sample_obs_shapes[key]
        
        obs_group.create_dataset(
            key,
            shape=obs_shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor
        )
    
    # Second pass: write data
    current_idx = 0
    for pkl_file in pkl_files:
        print(f"  Processing: {os.path.basename(pkl_file)}")
        with open(pkl_file, "rb") as f:
            transitions = pkl.load(f)
        
        if not isinstance(transitions, list):
            transitions = [transitions]
        
        n_trans = len(transitions)
        
        # Process in batches to reduce memory
        batch_size = min(1000, n_trans)
        for batch_start in range(0, n_trans, batch_size):
            batch_end = min(batch_start + batch_size, n_trans)
            batch = transitions[batch_start:batch_end]
            
            # Collect batch data
            batch_actions = []
            batch_obs = {key: [] for key in obs_keys}
            
            for trans in batch:
                # Actions
                if "actions" in trans:
                    action = np.array(trans["actions"], dtype=np.float32)
                else:
                    action = np.zeros(sample_action_shape, dtype=np.float32)
                batch_actions.append(action)
                
                # Observations - handle nested dicts
                if "observations" in trans:
                    flat_obs = flatten_obs_dict(trans["observations"])
                    for key in obs_keys:
                        if key in flat_obs:
                            obs = np.array(flat_obs[key])
                            while obs.ndim > 1 and obs.shape[0] == 1:
                                obs = obs.squeeze(0)
                        else:
                            obs = np.zeros(sample_obs_shapes[key])
                        
                        if len(sample_obs_shapes[key]) == 3:  # Image (H, W, C)
                            obs = obs.astype(np.uint8)
                        else:
                            obs = obs.astype(np.float32)
                        
                        batch_obs[key].append(obs)
            
            write_start = current_idx + batch_start
            write_end = current_idx + batch_end
            
            root["actions"][write_start:write_end] = np.array(batch_actions)
            for key in obs_keys:
                root["observations"][key][write_start:write_end] = np.array(batch_obs[key])
        
        current_idx += n_trans
        del transitions
        gc.collect()
    
    print(f"Conversion complete! Zarr store saved to: {zarr_path}")
    
    total_bytes = 0
    for key in root["observations"].array_keys():
        arr = root["observations"][key]
        total_bytes += arr.nbytes_stored
        ratio = arr.nbytes / arr.nbytes_stored if arr.nbytes_stored > 0 else 1
        print(f"  observations/{key}: {arr.nbytes_stored / 1e6:.1f} MB (compression ratio: {ratio:.1f}x)")
    
    arr = root["actions"]
    total_bytes += arr.nbytes_stored
    ratio = arr.nbytes / arr.nbytes_stored if arr.nbytes_stored > 0 else 1
    print(f"  actions: {arr.nbytes_stored / 1e6:.1f} MB (compression ratio: {ratio:.1f}x)")
    print(f"  Total on disk: {total_bytes / 1e6:.1f} MB")

def compute_files_hash(file_paths: List[str]) -> str:
    hash_input = ""
    for path in sorted(file_paths):
        mtime = os.path.getmtime(path)
        size = os.path.getsize(path)
        hash_input += f"{path}:{mtime}:{size};"
    
    return hashlib.md5(hash_input.encode()).hexdigest()[:12]

def flatten_obs_dict(obs_dict: dict) -> dict:
    result = {}
    
    for key, value in obs_dict.items():
        if isinstance(value, dict):
            if key == "state":
                state_arrays = []
                for sub_key, sub_value in value.items():
                    arr = np.array(sub_value).flatten()
                    state_arrays.append(arr)
                if state_arrays:
                    result["state"] = np.concatenate(state_arrays)
            else:
                for sub_key, sub_value in value.items():
                    result[sub_key] = sub_value
        else:
            result[key] = value
    
    return result

def compute_mc_returns(transitions, discount):
    mc_returns = np.zeros(len(transitions), dtype=np.float32)
    transitions_rewards = [t['rewards'] for t in transitions]
    transitions_masks = [t['masks'] for t in transitions]
    g = 0.0
    for t in reversed(range(len(transitions_rewards))):
        g = transitions_rewards[t] + discount * transitions_masks[t] * g
        mc_returns[t] = g

    for i, transition in enumerate(transitions):
        transition['mc_returns'] = mc_returns[i]
    return transitions