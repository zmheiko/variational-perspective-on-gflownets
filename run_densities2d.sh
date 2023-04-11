#!/bin/bash
set -ex
export CUDA_VISIBLE_DEVICES=
# export HYDRA_FULL_ERROR=1

mode=train
hostname=$(hostname)
experiment_name=experiment_densities2d
seeds=0
cvs=const_learned
datasets=2spirals
back_ratios=0
fixed_ebm=False

for cv in ${cvs//,/ }; do
    for dataset in ${datasets//,/ }; do
        for br in ${back_ratios//,/ }; do
            for seed in ${seeds//,/ }; do
                python3 experiment_densities2d.py -m\
                    args.exp_name=$experiment_name\
                    args.data=$dataset\
                    args.type=rklrf\
                    args.back_ratio=$br\
                    args.device=-1\
                    args.cv=$cv\
                    args.seed=$seed\
                    args.mode=$mode\
                    box.hostname=$hostname\
                    args.n_iters=1e5\
                    args.fixed_ebm=$fixed_ebm
            done
        done
    done
done
