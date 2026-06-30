import numpy as np

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