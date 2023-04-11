#!/bin/bash
set -ex
# export HYDRA_FULL_ERROR=1 

experiment_name=experiment_ising
hostname=$(hostname)
seeds=0
methods=virf
betas=1.0
# betas=-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0
cvs=opt
# cvs=lrn,log_Z,opt,opt_vec
mode=train
# mode=test

for seed in ${seeds//,/ }; do
    python3 experiment_ising.py -m\
        args.exp_name=$experiment_name\
        args.type=$methods\
        args.beta=$betas\
        args.N=15\
        args.device=0\
        args.cv=$cvs\
        args.seed=$seed\
        args.mode=$mode\
        box.hostname=$hostname\
        args.n_iters=5000\
        args.fixed_ebm=True\
        args.eval_every=100
done
