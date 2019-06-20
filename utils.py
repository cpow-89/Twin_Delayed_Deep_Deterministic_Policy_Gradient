from datetime import datetime
import os
import glob
import torch
import torch.nn.parameter


def mkdir(path):
    """if needed create a folder at given path"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def get_current_date_time():
    """get current datetime as string in the form of Y_m_d_H_M_S"""
    current_date_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    current_date_time = current_date_time.replace(" ", "__").replace("/", "_").replace(":", "_")
    return current_date_time


def save_state_dict(checkpoint_dir, state_dict):
    """Save the network weights"""
    mkdir(checkpoint_dir)
    current_date_time = get_current_date_time()
    torch.save(state_dict, os.path.join(checkpoint_dir, "ckpt_" + current_date_time))


def load_latest_available_state_dict(checkpoint_dir):
    list_of_files = glob.glob(checkpoint_dir)
    latest_file = max(list_of_files, key=os.path.getctime)
    return torch.load(latest_file)


def load_partial_state_dict(state_dict, target_state_dict):
    for name, param in state_dict.items():
        if name in target_state_dict:
            param = param.data
            target_state_dict[name].copy_(param)
    return state_dict
