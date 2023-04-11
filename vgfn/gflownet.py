import os, sys
import time
import copy
import random
from tqdm import tqdm

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists

from eb_gfn.network import make_mlp
from eb_gfn.gflownet import GFlowNet_LearnedPb_TB, GFlowNet_Randf_TB
from eb_gfn.gflownet import tb_mle_randf_loss as tb_mle_randf_loss_original

def get_GFlowNet(type, xdim, args, device, net=None):
    if type == "atbrf" or type == "tbrf":
        return GFlowNet_Randf_TB_mod(xdim=xdim, args=args, device=device, net=net)
    elif type == "rklrf" or type == "virf" or type=="aklrf":
        return GFlowNet_Randf_VI(xdim=xdim, args=args, device=device, net=net)
    else:
        raise NotImplementedError("No such model")

class GFlowNet_Randf_TB_mod(GFlowNet_Randf_TB):
    # binary data, train w/ long DB loss

    def train(self, batch_size, scorer, silent=False, data=None, back_ratio=0.,): #mle_coef=0., kl_coef=0., kl2_coef=0., pdb=False):
        # scorer: x -> logp
        assert self.train_steps == 1
        self.model.zero_grad()
        torch.cuda.empty_cache()

        gfn_loss, forth_loss, back_loss, mle_loss = \
            tb_mle_randf_loss_original(scorer, self, batch_size, back_ratio=back_ratio, data=data)
        gfn_loss, forth_loss, back_loss, mle_loss = \
            gfn_loss.mean(), forth_loss.mean(), back_loss.mean(), mle_loss.mean()

        loss = gfn_loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip > 0:
            breakpoint()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip, norm_type="inf")
        self.optimizer.step()

        return {
            "MLE": "{:.2e}".format(mle_loss.item()),
            "GFN": "{:.6e}".format(gfn_loss.item()),
            "RKL": "{:.6e}".format(forth_loss.item()),
            "FKL": "{:.6e}".format(back_loss.item()),
            # "log_Z": "{:.2e}".format(log_Z.item()),
            }

    def cal_logp(self, data, num: int):
        logp_ls = []
        for _ in range(num):
            mle_loss = cal_mle_randf_loss(self, data.shape[0], data=data)
            # mle_loss = lop_pf --> so there should not be a minus sign 
            # logpj = - mle_loss.detach().cpu() - torch.tensor(num).log()
            logpj = mle_loss.detach().cpu() - torch.tensor(num).log()
            logp_ls.append(logpj.reshape(logpj.shape[0], -1))

        batch_logp = torch.logsumexp(torch.cat(logp_ls, dim=1), dim=1)  # (bs,)
        return batch_logp.mean()

class GFlowNet_Randf_VI(GFlowNet_Randf_TB_mod):
    # binary data, train w/ long DB loss
    def __init__(self, xdim, args, device, net=None):
        # super().__init__()
        self.xdim = xdim
        self._hops = 0.
        # (bs, data_dim) -> (bs, data_dim)
        if net is None:
            self.model = make_mlp([xdim] + [args.hid] * args.hid_layers +
                              [3 * xdim], act=(nn.LeakyReLU() if args.leaky else nn.ReLU()), with_bn=args.gfn_bn)
        else:
            self.model = net
        self.model.to(device)

        self.device = device

        self.exp_temp = args.temp
        self.rand_coef = args.rand_coef  # involving exploration
        self.init_zero = args.init_zero
        self.clip = args.clip
        self.l1loss = args.l1loss

        self.replay = None
        self.tau = args.tau if hasattr(args, "tau") else -1

        self.train_steps = args.train_steps

        param_list = [{'params': self.model.parameters(), 'lr': args.glr}]
        
        # New attributes
        self.debug = True if args.debug else False
        self.cv_name = args.cv
        if args.cv == "const_learned":
            self.cv = nn.Parameter(torch.tensor(0., device=device))
            self.logZ = self.cv # compatability with original code
            param_list.append({'params': self.cv, 'lr': args.zlr})

        if args.opt == "adam":
            self.optimizer = torch.optim.Adam(param_list, weight_decay=args.gfn_weight_decay)
        elif args.opt == "sgd":
            self.optimizer = torch.optim.SGD(param_list, momentum=args.momentum, weight_decay=args.gfn_weight_decay)

    def get_logits(self, state):
        # Returns add_logits and del_logits
        logits = self.model(state) 
        if logits.isnan().any():
            breakpoint()
        return logits[:, :2 * self.xdim], logits[:, 2 * self.xdim:]

    def empty_trajectory(self, batch_size, data=None): 
        if self.init_zero:
            trajectory = torch.zeros((batch_size, self.xdim+1, self.xdim), device=self.device)
            state = torch.zeros((batch_size, self.xdim), device=self.device)
        else: # init -1
            trajectory = -1 * torch.ones((batch_size, self.xdim+1, self.xdim), device=self.device)
            state = -1 * torch.ones((batch_size, self.xdim), device=self.device)
        log_pf = torch.zeros((batch_size, self.xdim+1), device=self.device)
        log_pb = torch.zeros((batch_size, self.xdim+1), device=self.device)
        if data is not None:
            trajectory[:, -1] = data
        return trajectory, state, log_pf, log_pb 

    def state_mask(self, state, mask_null=False):
        # Return mask containing 1 at non-null locations
        if self.init_zero:
            if mask_null:
                mask = (state.abs() < 1e-8).float()
            else:
                mask = (state != 0).float() 
        else:
            if mask_null:
                mask = (state < -0.5).float()
            else:
                mask = (state > -0.5).float()
        return mask

    def indices_to_values(self, indices):
        if self.init_zero:
            values = 2 * indices - 1
        else:
            values = indices
        return values

    def sample_forward(self, batch_size, trajectory=None, debug_args=None):
        if trajectory is None:
            new_trajectory, state, log_pf, log_pb = self.empty_trajectory(batch_size)
        else:
            new_trajectory, state, log_pf, log_pb = self.empty_trajectory(batch_size)
            add_locs_trajectory = (trajectory[:, 1:] != trajectory[:, :-1]).nonzero()[:, -1].reshape(batch_size, self.xdim)
            add_values_trajectory = trajectory[:, 1:].gather(2, add_locs_trajectory.unsqueeze(2)).long()
            assert (trajectory != -1).any()

        for step in range(1, self.xdim+1):
            add_logits, _ = self.get_logits(state)
            # get locations
            mask = self.state_mask(state)
            add_probs = (1 - mask) / (1e-9 + (1 - mask).sum(1)).unsqueeze(1) # uniform prob over all null locations
            add_val_logits = add_logits.reshape(-1, self.xdim, 2) # [128, 32, 2]  
            if trajectory is None:
                add_locs = add_probs.multinomial(1) # [128, 1]
                log_add_val_probs = add_val_logits.gather(1, add_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze(1).log_softmax(1) # [128, 2]
                add_values = log_add_val_probs.exp().multinomial(1) # [128, 1]
            else:
                add_locs = add_locs_trajectory[:, step-1:step]
                log_add_val_probs = add_val_logits.gather(1, add_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze(1).log_softmax(1) # [128, 2]
                add_values = add_values_trajectory[:, step-1]

            if self.rand_coef > 0:
                updates = torch.bernoulli(self.rand_coef * torch.ones(x.shape[0])).int().to(x.device)
                add_values = (1 - add_values) * updates[:, None] + add_values * (1 - updates[:, None])
            
            state = state.scatter(1, add_locs, self.indices_to_values(add_values).float())
            new_trajectory[:, step] = state
            log_pf[:, step] = log_add_val_probs.gather(1, add_values).squeeze(1) - math.log(self.xdim - step + 1)
            log_pb[:, step] = -math.log(step)
        return new_trajectory, log_pf, log_pb

    def sample_backward(self, data):
        batch_size = data.shape[0]
        trajectory, _, log_pf, log_pb = self.empty_trajectory(batch_size, data=data)
        state = data

        for step in range(self.xdim+1, 0, -1):
            del_val_logits, _ = self.get_logits(state)

            if step < self.xdim + 1: # skip first (last) step
                del_val_logits = del_val_logits.reshape(-1, self.xdim, 2)
                log_del_val_prob = del_val_logits.gather(1, del_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze().log_softmax(1)
                log_pf[:, step] = log_del_val_prob.gather(1, deleted_val).squeeze(1) - math.log(self.xdim - step + 1)
                log_pb[:, step] = -math.log(step)
                trajectory[:, step-1] = state

            if step > 0: # skip last (first) step
                mask = self.state_mask(state, mask_null=True)
                del_locs = (0 - 1e9 * mask).softmax(1).multinomial(1)  # row sum not need to be 1
                deleted_val= state.gather(1, del_locs).long()
                del_values = torch.ones(batch_size, 1, device=self.device) * (0 if self.init_zero else -1)
                state = state.scatter(1, del_locs, del_values)
        return trajectory, log_pf, log_pb

    def train(self, batch_size, scorer, data=None, back_ratio=0): #mle_coef=0., kl_coef=0., kl2_coef=0., pdb=False):
        # scorer: x -> logp
        self.model.zero_grad()
        torch.cuda.empty_cache()

        gfn_loss, rkl_loss, fkl_loss, mle_loss = akl_randf_loss(
                scorer,
                self,
                batch_size,
                back_ratio=back_ratio,
                data=data,
                cv_name=self.cv_name,
                debug=self.debug,
                )
        self.optimizer.zero_grad()
        gfn_loss.backward()
        if self.clip > 0:
            breakpoint()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip, norm_type="inf")
        self.optimizer.step()
        return {
            "aKL": "{:.6e}".format(gfn_loss.item()),
            "RKL": "{:.6e}".format(rkl_loss.item()),
            "FKL": "{:.6e}".format(fkl_loss.item()),
            "MLE": "{:.6e}".format(mle_loss.item()),
            }

def akl_randf_loss(scorer, gfn, batch_size, back_ratio=0., data=None, cv_name=None, debug=False):
            rkl_loss, fkl_loss, mle_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            if back_ratio < 1.:
                if debug:
                    rng_state_before_forward = torch.get_rng_state()
                trajectory_fwd, log_pf_fwd, log_pb_fwd = gfn.sample_forward(batch_size)
                score_value_fwd, log_pf_fwd, log_pb_fwd = scorer(trajectory_fwd[:, -1]), log_pf_fwd.sum(dim=-1), log_pb_fwd.sum(dim=-1)
                log_Z = gfn.logZ if cv_name=="const_learned" else log_Z(log_pb_fwd + score_value_fwd - log_pf_fwd)
                rkl_loss = rkl_grad_estimator(
                        gfn, 
                        trajectory_fwd,
                        log_pf_fwd, 
                        log_pb_fwd,
                        score_value_fwd,
                        cv_name=cv_name,
                        )
                rkl_loss = rkl_loss.mean()

            if back_ratio > 0.:
                if debug:
                    rng_state_before_backward = torch.get_rng_state()
                trajectory_bwd, log_pf_bwd, log_pb_bwd = gfn.sample_backward(data)
                score_value_bwd, log_pf_bwd, log_pb_bwd = scorer(data), log_pf_bwd.sum(dim=-1), log_pb_bwd.sum(dim=-1)

                fkl_loss = fkl_grad_estimator(log_pf_bwd, log_pb_bwd, score_value_bwd)
                fkl_loss = fkl_loss.mean()
                mle_loss = log_pf_bwd.mean()

            gfn_loss = (1 - back_ratio) * rkl_loss + back_ratio * fkl_loss
            return gfn_loss, rkl_loss, fkl_loss, mle_loss

def tb_mle_randf_loss(ebm_model, gfn, batch_size, back_ratio=0., data=None, log_Z=None, debug=False):
    assert log_Z is not None
    if back_ratio < 1.:
        if gfn.init_zero:
            x = torch.zeros((batch_size, gfn.xdim)).to(gfn.device)
            trajectory= torch.zeros((batch_size, gfn.xdim+1, gfn.xdim)).to(gfn.device)
        else:
            x = -1 * torch.ones((batch_size, gfn.xdim)).to(gfn.device)
            trajectory= -1 * torch.ones((batch_size, gfn.xdim + 1, gfn.xdim)).to(gfn.device)

        log_pf = 0.
        # debug outputs
        log_pf_j = torch.zeros((batch_size, gfn.xdim)).to(gfn.device)
        add_values_j = torch.zeros((batch_size, gfn.xdim, 1)).to(gfn.device)
        add_val_prob_j = torch.zeros((batch_size, gfn.xdim, 2)).to(gfn.device)
        debug_outs = (trajectory, log_pf_j, add_val_prob_j, add_values_j)
        for step in range(gfn.xdim + 1):
            logits = gfn.model(x)
            add_logits, _ = logits[:, :2 * gfn.xdim], logits[:, 2 * gfn.xdim:]

            if step < gfn.xdim:
                # mask those that have been edited
                if gfn.init_zero:
                    mask = (x != 0).float()
                else:
                    mask = (x > -0.5).float()
                add_prob = (1 - mask) / (1e-9 + (1 - mask).sum(1)).unsqueeze(1)
                add_locs = add_prob.multinomial(1)

                add_val_logits = add_logits.reshape(-1, gfn.xdim, 2)
                add_val_prob = add_val_logits.gather(1, add_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze().softmax(1)
                add_values = add_val_prob.multinomial(1)

                if gfn.rand_coef > 0:
                    # updates = torch.distributions.Bernoulli(probs=gfn.rand_coef).sample(sample_shape=torch.Size([x.shape[0]]))
                    updates = torch.bernoulli(gfn.rand_coef * torch.ones(x.shape[0])).int().to(x.device)
                    add_values = (1 - add_values) * updates[:, None] + add_values * (1 - updates[:, None])

                log_pf = log_pf + add_val_prob.log().gather(1, add_values).squeeze(1)  # (bs, 1) -> (bs,)
                # debug outputs
                log_pf_j[:, step] = add_val_prob.log().gather(1, add_values).squeeze(1)
                add_val_prob_j[:, step] = add_val_prob
                add_values_j[:, step] = add_values

                if gfn.init_zero:
                    add_values = 2 * add_values - 1

                x = x.scatter(1, add_locs, add_values.float())
                trajectory[:, step+1] = x
        assert torch.all(x != 0) if gfn.init_zero else torch.all(x >= 0)

        score_value = ebm_model(x)
        if gfn.l1loss:
            breakpoint()
            forth_loss = F.smooth_l1_loss(log_Z + log_pf - score_value, torch.zeros_like(score_value))
        else:
            forth_loss = (log_Z + log_pf - score_value) ** 2
    else:
        forth_loss = torch.tensor(0.).to(gfn.device)

    # traj is from given data back to s0, sample w/ unif back prob
    mle_loss = torch.tensor(0.).to(gfn.device)
    if back_ratio <= 0.:
        back_loss = torch.tensor(0.).to(gfn.device)
    else:
        assert data is not None
        if gfn.init_zero:
            trajectory_bwd = torch.zeros((batch_size, gfn.xdim+1, gfn.xdim)).to(gfn.device)
        else:
            trajectory_bwd = -1 * torch.ones((batch_size, gfn.xdim + 1, gfn.xdim)).to(gfn.device)
        trajectory_bwd[:, gfn.xdim] = data
        x = data
        batch_size = x.size(0)
        back_loss = torch.zeros(batch_size).to(gfn.device)

        for step in range(gfn.xdim + 1):
            logits = gfn.model(x)
            del_val_logits, _ = logits[:, :2 * gfn.xdim], logits[:, 2 * gfn.xdim:]

            if step > 0:
                del_val_logits = del_val_logits.reshape(-1, gfn.xdim, 2)
                log_del_val_prob = del_val_logits.gather(1, del_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze().log_softmax(1)
                mle_loss = mle_loss + log_del_val_prob.gather(1, deleted_val).squeeze(1)

            if step < gfn.xdim:
                if gfn.init_zero:
                    mask = (x.abs() < 1e-8).float()
                else:
                    mask = (x < -0.5).float()
                del_locs = (0 - 1e9 * mask).softmax(1).multinomial(1)  # row sum not need to be 1
                deleted_val = x.gather(1, del_locs).long()
                del_values = torch.ones(batch_size, 1).to(gfn.device) * (0 if gfn.init_zero else -1)
                x = x.scatter(1, del_locs, del_values)
                trajectory_bwd[:, gfn.xdim - step - 1] = x
        debug_outs = (trajectory_bwd, mle_loss)

    # if back_ratio > 0.:
    if gfn.l1loss:
        back_loss = F.smooth_l1_loss(log_Z + mle_loss - ebm_model(data).detach(), torch.zeros_like(mle_loss))
    else:
        back_loss = (log_Z + mle_loss - ebm_model(data).detach()) ** 2

    gfn_loss = (1 - back_ratio) * forth_loss + back_ratio * back_loss
    return gfn_loss, forth_loss, back_loss, mle_loss, debug_outs


def cal_mle_randf_loss(gfn, batch_size, data=None):
    # traj is from given data back to s0, sample w/ unif back prob
    mle_loss = torch.tensor(0.).to(gfn.device)
    assert data is not None
    x = data
    batch_size = x.size(0)

    for step in range(gfn.xdim + 1):
        logits = gfn.model(x)
        del_val_logits, _ = logits[:, :2 * gfn.xdim], logits[:, 2 * gfn.xdim:]

        if step > 0:
            del_val_logits = del_val_logits.reshape(-1, gfn.xdim, 2)
            log_del_val_prob = del_val_logits.gather(1, del_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze().log_softmax(1)
            mle_loss = mle_loss + log_del_val_prob.gather(1, deleted_val).squeeze(1)

        if step < gfn.xdim:
            if gfn.init_zero:
                mask = (x.abs() < 1e-8).float()
            else:
                mask = (x < -0.5).float()
            del_locs = (0 - 1e9 * mask).softmax(1).multinomial(1)  # row sum not need to be 1
            deleted_val = x.gather(1, del_locs).long()
            del_values = torch.ones(batch_size, 1).to(gfn.device) * (0 if gfn.init_zero else -1)
            x = x.scatter(1, del_locs, del_values)

    return mle_loss

def eval0(e):
    return e - e.detach()

def eval1(e):
    return torch.exp(eval0(e))

def log_Z(lw):
    return torch.logsumexp(lw, dim=0) - math.log(lw.shape[0])

def get_control_variate(gfn, cv_name, trajectory, log_w):
    cv = None
    if cv_name == "const_learned":
        cv = gfn.cv
    elif cv_name == "E_log_w":
        cv = loo_E_log_w(log_w).detach()
    elif cv_name == "loo_log_Z":
        cv = loo_log_Z(log_w).detach()
    elif cv_name == "loo_opt":
        cv = loo_opt_cv(gfn, trajectory, log_w).detach()
    else:
        raise NotImplementedError(f"Control Variate with name \"{cv_name}\" not implemented")
    return cv 

def fkl_grad_estimator(log_pf, log_pb, score_value, log_Z=None):
    fkl_loss = log_pb.detach() + score_value.detach() - log_pf
    return fkl_loss

def rkl_grad_estimator(gfn, trajectory, log_pf, log_pb, score_value, cv_name=None):
    log_w = log_pb + score_value - log_pf
    cv = get_control_variate(gfn, cv_name, trajectory, log_w)
    f = -log_w + cv
    rkl_loss = eval1(f) * f.detach() # --grad-> 1 * 0 + gf * f
    if rkl_loss.isnan().any():
        breakpoint()
    return rkl_loss
