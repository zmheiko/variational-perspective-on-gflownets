from pathlib import Path
import datetime
import numpy as np
import torch
import torch.nn as nn
import os, sys

import time
import random
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
# import argparse
import logging

from vgfn.gflownet import get_GFlowNet
from vgfn.utils import save_model, load_model, set_seed, save_data
import vgfn.utils

from eb_gfn.synthetic.synthetic_utils import plot_heat, plot_samples,\
    float2bin, bin2float, get_binmap, get_true_samples, get_ebm_samples, EnergyModel
from eb_gfn.synthetic.synthetic_data import inf_train_gen, OnlineToyDataset
from eb_gfn.synthetic.synthetic_utils import exp_hamming_mmd

def init_env(cfg):
    set_seed(cfg.args.seed)

def init_model(cfg, path, dim=32, checkpoint=None, device='cpu'):
    if checkpoint is None:
        checkpoint = int(cfg.args.n_iters-1)
    gfn = get_GFlowNet(cfg.args.type, dim, cfg.args, device)
    ebm = EnergyModel(dim, 256).to(device)
    gfn_dict = {f"gfn": gfn.model}
    ebm_dict = {f"ebm": ebm}
    if cfg.args.fixed_ebm:
        ebm_fixed_dict = {cfg.args.data: ebm}
        load_model(ebm_fixed_dict, path=os.path.join(path, "../../../../densities2d_ebms_fixed"))
    else: 
        load_model(ebm_dict, path=os.path.join(path, "models"), suffix=checkpoint)
    load_model(gfn_dict, path=os.path.join(path, "models"), suffix=checkpoint)
    model_dict = {**ebm_dict, **gfn_dict}
    model_dict["gfn"] = gfn
    return model_dict

def load_data(cfg, path, checkpoint=None):
    if checkpoint is None:
        checkpoint = int(cfg.args.n_iters-1)
    return vgfn.utils.load_data("evaluation", path=os.path.join(path, "data"), suffix=checkpoint)

def back_ratio_schedule(itr, itr_max):
    switch = itr_max // 2 + 1
    if itr < switch:
        return 1.
    else:
        return 1 - ((itr - switch) / switch)

unif_dist = torch.distributions.Bernoulli(probs=0.5)

log = logging.getLogger(__name__)
@hydra.main(config_path="config", config_name="densities2d")
def run_experiment(cfg : DictConfig) -> None:
    args = cfg.args
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
    device = torch.device("cpu") if args.device < 0 else torch.device("cuda")
    save_dir = os.getcwd()

    log.info("Device:" + str(device))
    log.info("args:" + str(args))

    ############## Data
    discrete_dim = 32
    bm, inv_bm = get_binmap(discrete_dim, 'gray')

    db = OnlineToyDataset(args.data, discrete_dim)
    if not hasattr(args, "int_scale"):
        int_scale = db.int_scale
    else:
        int_scale = args.int_scale
    if not hasattr(args, "plot_size"):
        plot_size = db.f_scale
    else:
        db.f_scale = args.plot_size
        plot_size = args.plot_size
    # plot_size = 4.1

    batch_size = args.batch_size
    multiples = {'pinwheel': 5, '2spirals': 2}
    batch_size = batch_size - batch_size % multiples.get(args.data, 1)

    ############## EBM 
    energy_model = EnergyModel(discrete_dim, 256).to(device)
    if cfg.args.fixed_ebm:
        load_model({args.data: energy_model}, path=os.path.join(hydra.utils.get_original_cwd(), "densities2d_ebms"))
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=args.lr)
    log.info("model: {:}".format(energy_model))

    ############## GFN model
    assert args.gmodel == "mlp"
    xdim = discrete_dim
    gfn = get_GFlowNet(args.type, xdim, args, device)

    if args.mode == "train":
        log.info(">>> TRAINING MODEL")
        itr = 0
        best_val_ll = -np.inf
        best_itr = -1
        lr = args.lr
        save_model({f"gfn": gfn.model, f"ebm": energy_model, f"opt_ebm": optimizer, f"opt_gfn": gfn.optimizer}, suffix=0)
        # while itr < args.n_iters:
        pbar = tqdm(range(int(args.n_iters)))
        for itr in pbar:            
            st = time.time()
            x = get_true_samples(db, batch_size, bm, int_scale, discrete_dim).to(device)

            update_success_rate = -1.
            ### Train GFN
            gfn.model.train()
            back_ratio=args.back_ratio if args.back_ratio != "scheduled" else back_ratio_schedule(itr, args.n_iters)
            train_dict = gfn.train(batch_size, 
                    scorer=lambda inp: energy_model(inp).detach(), 
                    data=x, 
                    back_ratio=back_ratio,
                    )
            pbar.set_postfix(train_dict)
            ### Get samples for CD method 
            if args.rand_k or args.lin_k or (args.K > 0):
                if args.rand_k:
                    K = random.randrange(xdim) + 1
                elif args.lin_k:
                    K = min(xdim, int(xdim * float(itr + 1) / args.warmup_k))
                    K = max(K, 1)
                elif args.K > 0:
                    K = args.K
                else:
                    raise ValueError

                gfn.model.eval()
                x_fake, delta_logp_traj = gfn.backforth_sample(x, K)

                delta_logp_traj = delta_logp_traj.detach()
                if args.with_mh:
                    # MH step, calculate log p(x') - log p(x)
                    lp_update = energy_model(x_fake).squeeze() - energy_model(x).squeeze()
                    update_dist = torch.distributions.Bernoulli(logits=lp_update + delta_logp_traj)
                    updates = update_dist.sample()
                    x_fake = x_fake * updates[:, None] + x * (1. - updates[:, None])
                    update_success_rate = updates.mean().item()

            else:
                x_fake = gfn.sample(batch_size)

        
            if itr % args.ebm_every == 0:
                ### Train ebm with CD method
                st = time.time() - st

                energy_model.train()
                logp_real = energy_model(x).squeeze()

                logp_fake = energy_model(x_fake).squeeze()
                obj = logp_real.mean() - logp_fake.mean()
                if not args.fixed_ebm:
                    # assert False
                    loss = -obj
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            if itr % args.print_every == 0 or itr == args.n_iters - 1:
                log.info("({:5d}) | ({:.3f}s/iter) cur lr= {:.2e} |log p(real)={:.2e}, "
                        "log p(fake)={:.2e}, diff={:.2e}, mh_acceptace_rate={:.1f}".format(
                    itr, st, lr, logp_real.mean().item(), logp_fake.mean().item(), obj.item(), update_success_rate))


            if (itr + 1) % args.eval_every == 0:
                # heat map of energy
                save_model({
                    f"gfn": gfn.model, 
                    f"ebm": energy_model,
                    f"opt_ebm": optimizer, 
                    f"opt_gfn": gfn.optimizer,
                    },
                    suffix=itr)
                gfn_test_ll, gfn_test_mmd = eval_model(gfn, energy_model, save_dir, args, device,
                    plot_size, inv_bm, int_scale, discrete_dim, itr, db, bm)
                log.info(f"Test GFN NLL ({itr}): {-gfn_test_ll.item():.3f}")
                log.info(f"Test GFN MMD ({itr}): {-gfn_test_mmd.item():.3f}")

            itr += 1
            if itr > args.n_iters:
                quit(0)

    elif args.mode == "test":
        log.info(">>> TESTING MODEL")
        max_itr = int(args.n_iters-1)
        load_model({
            "gfn": gfn.model, 
            "ebm": energy_model
            },
            suffix=int(args.n_iters)-1)
        gfn_test_ll, gfn_test_mmd = eval_model(gfn, energy_model, save_dir, args, device,
            plot_size, inv_bm, int_scale, discrete_dim, args.n_iters-1, db, bm, suffix="test")
        log.info(f"Test GFN NLL ({max_itr}): {-gfn_test_ll.item():.3f}")
        log.info(f"Test GFN MMD ({max_itr}): {-gfn_test_mmd.item():.3f}")
    else:
        raise NotImplementedError("Can run in train or test model only")

def plot(save_dir, ebm, inv_bm, bm, plot_size, device, 
        int_scale, discrete_dim, gfn_samples, suffix=None):
    plot_dir = os.path.join(save_dir, "plots")
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    suffix = f"_{suffix}" if suffix is not None else ""

    plot_heat(ebm, bm, plot_size, device, int_scale, discrete_dim,
            out_file=os.path.join(plot_dir, f'heat{suffix}.pdf'))
    gfn_samp_float = bin2float(gfn_samples.data.cpu().numpy().astype(int), 
            inv_bm, int_scale, discrete_dim)
    plot_samples(gfn_samp_float, os.path.join(plot_dir, f'gfn_samples{suffix}.pdf'), lim=plot_size)

def eval_model(gfn, ebm, save_dir, args, device,
        plot_size, inv_bm, int_scale, discrete_dim, itr, db, bm, suffix=None):
    test_batch_size = 4000

    gfn_samples = gfn.sample(test_batch_size).detach()
    score_value = ebm(gfn_samples)
     
    plot(save_dir, ebm, inv_bm, bm, plot_size, device, 
            int_scale, discrete_dim, gfn_samples, 
            suffix=itr if suffix is None else f"{itr}_{suffix}")
    true_samples = get_true_samples(db, test_batch_size, bm, int_scale, discrete_dim)
    
    # GFN MMD
    gfn_test_mmd = exp_hamming_mmd(gfn_samples, true_samples.to(gfn_samples.device))

    # GFN LL
    gfn.model.eval()
    logps = []
    pbar = tqdm(range(10))
    pbar.set_description("GFN Calculating likelihood")
    for _ in pbar:
        pos_samples_bs = get_true_samples(db, 1000, bm, int_scale, discrete_dim).to(device)
        logp = gfn.cal_logp(pos_samples_bs, 20)
        logps.append(logp.reshape(-1))
        pbar.set_postfix({"logp": f"{torch.cat(logps).mean().item():.2f}"})
    gfn_test_ll = torch.cat(logps).mean()

    save_data({
            "batch_size": test_batch_size,
            # "log_Z": log_Z if args.type == "rklrf" else torch.tensor(torch.nan),
            "ll": gfn_test_ll,
            "mmd": gfn_test_mmd,
        },
        name="evaluation",
        suffix=[itr, suffix],
        )
    return gfn_test_ll, gfn_test_mmd

if __name__ == "__main__":
    run_experiment()
