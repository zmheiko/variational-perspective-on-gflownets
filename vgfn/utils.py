import os
import torch
from pathlib import Path

def save_model(model_dict, path='./models', suffix=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    for name, model in model_dict.items():
        # torch.save(model, '%s/%s'%(path, name))
        torch.save(model.state_dict(), os.path.join(path, mk_filename(name, suffix)))

def load_model(model_dict, path='./models', suffix=None):
    for name, model in model_dict.items():
        model.load_state_dict(torch.load(os.path.join(path, mk_filename(name, suffix))))
        model.eval()

def load_data(name, path='./data', suffix=None):
    a = torch.load(os.path.join(path, mk_filename(name, suffix)))
    if a is None:
        breakpoint()
    return a

def save_data(data, name, path='./data', suffix=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    torch.save(data, os.path.join(path, mk_filename(name, suffix)))

def mk_filename(name, suffix, ft="pt"):
    if isinstance(suffix, list):
        suffix = "_".join([str(s) for s in suffix if s is not None])

    if suffix is not None:
        suffix = "_" + str(suffix)
    else:
       suffix = ""

    filename = "%s%s.%s"%(name, suffix, ft)   
    return filename

def set_seed(seed):
    torch.manual_seed(seed)

def add_if_attr(dict, model, name, add_name=None):
    add_name = name if add_name is None else add_name
    value = getattr(model, name, None)
    if value is not None:
        dict[add_name] = value.detach()
    
