from typing import Union

from omegaconf import OmegaConf
import json
import pickle
import torch


def load_yaml(file_path: str):
    if isinstance(file_path, str) and file_path.endswith("yaml"):
        config = OmegaConf.load(file_path)
    else:
        raise ValueError("The input [file_path] is not the path of a <yaml> file.")
        
    return config


def load_json(file_path: str):
    if isinstance(file_path, str) and file_path.endswith("json"):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        raise ValueError("The input [file_path] is not the path of a <json> file.")
    
    return data


def save_json(file_path: str, data: Union[dict, list]):
    if isinstance(file_path, str) and file_path.endswith("json"):
        with open(file_path, 'a') as f:
            json_str = json.dumps(data)
            f.write(json_str + '\n')
    else:
        raise ValueError("The input [file_path] is not the path of a <json> file.")
    
    return


def load_pkl(file_path: str):
    if isinstance(file_path, str) and file_path.endswith("pkl"):
        with open(file_path, "rb") as file:
            data = pickle.load(file)
    else:
        raise ValueError("The input [file_path] is not the path of a <pkl> file.")
        
    return data


def save_pkl(file_path: str, data):
    if isinstance(file_path, str):
        if not file_path.endswith("pkl"):
            file_path += '.pkl'
        with open(file_path, "wb") as file:
            pickle.dump(data, file)
    else:
        raise ValueError("The input [file_path] is not of type <str>.")
        
    return
