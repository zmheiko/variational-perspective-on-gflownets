import os
from pathlib import Path
import datetime
import jax
import jax.numpy as jnp
from jax import random
from jax.flatten_util import ravel_pytree
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax import distributions
import optax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
# import argparse
import logging

import vgfn.gflownet_jax as gfn
import vgfn.utils_jax as utils
import vgfn.ising as ebm


def load_params(cfg, path="./", dir="./params", suffix=""):
    path = os.path.join(path, dir) 
    gfn_params = dict(utils.load_params("gfn_params", path=path, suffix=suffix))
    ebm_params = dict(utils.load_params("ebm_params", path=path, suffix=suffix)) 
    opt_state = dict(utils.load_params("opt_state", path=path, suffix=suffix))
    return {"gfn_params": gfn_params, "ebm_params": ebm_params, "opt_state": opt_state}

def load_env(cfg, path="./", dir="./env", suffix=""):
    pass

def load_data(cfg, path="./", dir="./eval", suffix=""):
    path = os.path.join(path, dir)
    data_dict = dict(utils.load_params("evaluation", path=path, suffix=suffix)) 
    return data_dict


log = logging.getLogger(__name__)
@hydra.main(config_path="config", config_name="ising", version_base="1.1")
def run_experiment(cfg : DictConfig) -> None:
    args = cfg.args
    rng_key = random.PRNGKey(args.seed)

    if args.mode == "train":
        log.info(">>> TRAINING MODEL")

        rng_key, gfn_params = gfn.init_params(rng_key, args)
        rng_key, ebm_params = ebm.init_params(rng_key, args)
        # optimizer = optax.adam(args.lr)
        optimizer = optax.multi_transform({"wnb": optax.adam(args.lr), "cv": optax.adam(args.zlr)}, 
                param_labels=lambda param_dict: {k: k for k, v in param_dict.items()})
        opt_state = optimizer.init(gfn_params)

        utils.save_params(name="gfn_params", path="./params", suffix=0, **gfn_params)
        utils.save_params(name="ebm_params", path="./params", suffix=0, **ebm_params)
        utils.save_params(name="opt_state", path="./params", suffix=0, opt_state=opt_state)
        
        pbar = tqdm(range(int(args.n_iters)))
        for itr in pbar:            
            rng_key, rng_key_loss = random.split(rng_key, 2)
            if args.cv == "lrn":
                loss, grad = jax.value_and_grad(gfn.loss_fn, argnums=3)(
                        rng_key_loss, gfn, ebm, gfn_params, ebm_params,
                        args.N**2, args.back_ratio, args.batch_size, args.init_zero, cv_name=args.cv)

            else:
                loss, grad, grad_vanilla = gfn.grad_loss_fn(
                        rng_key_loss, gfn, ebm, gfn_params, ebm_params,
                        args.N**2, args.back_ratio, args.batch_size, args.init_zero, cv_name=args.cv)
                if args.debug:
                    loss_test, grad_vanilla_test = jax.value_and_grad(gfn.loss_fn, argnums=3)(
                            rng_key_loss, gfn, ebm, gfn_params, ebm_params,
                            args.N**2, args.back_ratio, args.batch_size, args.init_zero, cv_name="none")
                    assert jnp.equal(loss, loss_test)
                    assert ravel_pytree(jax.tree_map(lambda x, y: jnp.isclose(x, y, atol=1e-2).all(), 
                                                    grad_vanilla, grad_vanilla_test))[0].all()

            updates, opt_state = optimizer.update(grad, opt_state)
            gfn_params = optax.apply_updates(gfn_params, updates)
            pbar.set_postfix({"loss": loss})

            if (itr + 1) % args.eval_every == 0 or (itr + 1) == args.n_iters:
                rng_key, rng_key_eval = random.split(rng_key, 2)
                utils.save_params(name="gfn_params", path="./params", suffix=itr, **gfn_params)
                utils.save_params(name="ebm_params", path="./params", suffix=itr, **ebm_params)
                utils.save_params(name="opt_state", path="./params", suffix=itr, opt_state=opt_state)
                
                eval_dict = eval_model(rng_key_eval, gfn_params, ebm_params, args, suffix=itr)
                log.info(f"Test GFN E[log w] ({itr}): {eval_dict['log_Z']:.3f}")
                log.info(f"Test GFN log Z ({itr}): {eval_dict['E_log_w']:.3f}")

    elif args.mode == "test":
        log.info(">>> TESTING MODEL")
        param_dict = load_params(cfg, suffix=args.n_iters-1)
        eval_dict = eval_model(rng_key, param_dict["gfn_params"], param_dict["ebm_params"], args, suffix=[args.n_iters-1, "test"])
        log.info(f"Test GFN E[log w] ({args.n_iters-1}): {eval_dict['log_Z']:.3f}")
        log.info(f"Test GFN log Z ({args.n_iters-1}): {eval_dict['E_log_w']:.3f}")
    else:
        raise NotImplementedError("Possible modes: train/test")

    rng_key, rng_key_plot = random.split(rng_key, 2)
    print(">>> PLOTTING", end = "...")
    plot_gfn_sampels_vs_ground_truth(rng_key_plot, gfn, gfn_params, args.N, args.init_zero, beta=args.beta)
    print("done!")

def plot_gfn_sampels_vs_ground_truth(rng_key, gfn, gfn_params, N, init_zero, beta):
    M = 5
    grids, _, _ = gfn.vmapped_sample_forward(rng_key, M, gfn_params, dim=N**2, init_zero=True)
    grids = grids.reshape(M, N, N) 
    grids_mh, ar = ebm.IsingModel(N, beta=beta).simulate_mh(100000, num_chains=M, return_chains=False)

    fig = plt.figure(figsize=(9, 3), dpi=300) 
    gs = gridspec.GridSpec(2, M, wspace=0.07, hspace=0.05)
    for i in range(M):
        ax = plt.subplot(gs[0, i])
        ax.matshow(grids_mh[i], vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2 = plt.subplot(gs[1, i])
        ax2.matshow(grids[i], vmin=-1, vmax=1)
        ax2.set_xticks([])
        ax2.set_yticks([])
        if i == 0:
            ax.set_ylabel("MH", fontsize=14)
            ax2.set_ylabel("GFN", fontsize=14)
    Path("./plots").mkdir(exist_ok=True)
    plt.savefig("./plots/samples.png")

def eval_model(rng_key, gfn_params, ebm_params, args, suffix=None):
    test_batch_size = 4000
    samples, log_pf, log_pb = gfn.vmapped_sample_forward(rng_key, test_batch_size, gfn_params, args.N**2, args.init_zero)
    score_value= ebm.vmapped_model(ebm_params, samples)
    log_w = log_pf.sum(axis=-1) + score_value - log_pb.sum(axis=-1)
    E_log_w = log_w.mean()
    log_Z = gfn.log_Z(log_w)
    eval_dict = {
            "batch_size": test_batch_size,
            "E_log_w": E_log_w,
            "log_Z": log_Z,
            "avg_score": score_value.mean(),
            "avg_log_pf": log_pf.sum(axis=-1).mean(),
            "avg_log_pb": log_pb.sum(axis=-1).mean(),
        }
    utils.save_params(name="evaluation", path="./eval", suffix=suffix, **eval_dict)
    return eval_dict

if __name__ == "__main__":
    run_experiment()
