import os
import jax.numpy as jnp 
from pathlib import Path
from collections.abc import Iterable 
import pickle
import jax.tree_util as tree_util
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

def save_params(name, path="./", suffix=None, **params):
    Path(path).mkdir(parents=True, exist_ok=True)
    # jnp.savez(os.path.join(path, mk_filename(name, suffix)), **params)
    with open(os.path.join(path, mk_filename(name, suffix, ft="pkl")), 'wb') as file:
        pickle.dump(params, file)

def load_params(name, path='./', suffix=None):
    with open(os.path.join(path, mk_filename(name, suffix, ft="pkl")), 'rb') as file:
        params = pickle.load(file)
    return params

def mk_filename(name, suffix, ft="npz"):
    if isinstance(suffix, Iterable):
        suffix = "_".join([str(s) for s in suffix if s is not None])
    if suffix is not None:
        suffix = "_" + str(suffix)
    else:
       suffix = ""

    filename = "%s%s.%s"%(name, suffix, ft)   
    return filename 

def is_experiment_node(x):
    return "valid" in x

def is_seed_node(x):
   return "valid" in x[list(x.keys())[0]] 

def map_experiments(map, experiments):
    return tree_util.tree_map(lambda x: map(x) if x["valid"] else jnp.array(jnp.nan), experiments, 
                              is_leaf=is_experiment_node)

def map_stack_experiments(map, experiments):
    mapped_dict = map_experiments(map, experiments)
    return tree_util.tree_map(lambda x: jnp.stack(list(x.values())), mapped_dict, 
                              is_leaf=lambda x: not isinstance(x[list(x.keys())[0]], dict))

def map_stack_map_experiments(map1, map2, experiments):
    stacked_dict = map_stack_experiments(map1, experiments)
    return tree_util.tree_map(map2, stacked_dict)

def valid_experiments(experiments):
    return tree_util.tree_map(lambda x: [k for k, y in x.items() if y["valid"]], experiments, is_leaf=is_seed_node)

def mk_kwarg_tree(kwargs_dict, kwarg_path={}):
    if kwargs_dict:
        key = list(kwargs_dict.keys())[0]
        key_values = kwargs_dict[key]
        key_values = key_values if isinstance(key_values, list) else [key_values]

        kwarg_tree = {}
        for value in key_values:
            kwarg_tree[value] = mk_kwarg_tree(kwargs_dict={k: v for k, v in kwargs_dict.items() if k!=key}, kwarg_path={**kwarg_path, key: value})
        return kwarg_tree 
    else:
        return {"kwargs": kwarg_path}

def load_experiments(kwargs_dict, override_fn, *args, **kwargs):
        experiments_kwarg_tree = mk_kwarg_tree(kwargs_dict, kwarg_path={})  
        return tree_util.tree_map(lambda x: load_experiment(override_fn(**x["kwargs"]), *args, **kwargs),
                                  experiments_kwarg_tree,
                                  is_leaf=lambda x: "kwargs" in x)

def download_rsync(user, server, path, to_path="./", ignore_existing=False, silent=True):
    try:
        Path(to_path).mkdir(parents=True, exist_ok=True)
        options = ["-vvvrP"]
        if ignore_existing:
            options.append("--ignore-existing")
        subprocess.call([
            "rsync",
            *options,
            f"{user}@{server}:{path}/",
            f"{to_path}"
            ], 
            stdout=subprocess.DEVNULL if silent else subprocess.STDOUT) 
    except:
        raise FileNotFoundError()

def load_experiment(
        overrides,
        load_params,
        load_env=None,
        load_data=None,
        base_dir=None,
        conf_dir="./conf",
        config_name="config",
        host_dict=None, 
        ignore_existing=False,
        version_base="1.1",
        ):
    # Reinitialize to prevent error if already initialized - e.g. rerunning cell in notebook
    GlobalHydra.instance().clear()
    base_dir = os.getcwd() if base_dir is None else base_dir
    rel_dirpath = os.path.relpath(os.path.abspath(base_dir),
                                  os.path.dirname(os.path.abspath(__file__)))
    initialize(config_path=os.path.join(rel_dirpath, conf_dir), version_base=version_base)
    cfg = compose(config_name=config_name, return_hydra_config=True, overrides=overrides)
    rel_experiment_dir = os.path.join(cfg.hydra.sweep.dir, cfg.hydra.sweep.subdir)
    experiment_dir = os.path.join(base_dir, rel_experiment_dir)

    experiment_dict = {
            "cfg": cfg,
            "path": experiment_dir,
            }
    # setup
    host_dict = {"local": None, **host_dict} if host_dict is not None else {"local": None}
    for h, (hostname, host) in enumerate(host_dict.items()):
        try:
            if hostname != "local":
                download_rsync(host["user"], 
                        host["server"], 
                        os.path.join(host["base_dir"], rel_experiment_dir),
                        experiment_dir,
                        ignore_existing=ignore_existing)

            os.chdir(experiment_dir)
            param_dict = load_params(cfg, path=experiment_dir, suffix=cfg.args.n_iters-1) 
            experiment_dict["params"] = param_dict
            if load_data is not None:
                experiment_dict["eval"] = load_data(cfg, path=experiment_dir, suffix=cfg.args.n_iters-1)
            if load_env is not None:
                load_env(cfg, path=experiment_dir, suffix=cfg.args.n_iters-1)
            # It worked, clean up and leave
            os.chdir(base_dir)
            break # it worked
        except FileNotFoundError as e:
            if h == len(host_dict) - 1:
                return{**experiment_dict, "valid": False, "error": e}
        finally:
            os.chdir(base_dir)

    return{**experiment_dict, "valid": True}
