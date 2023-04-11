import os, sys
import time
import copy
from functools import partial
from tqdm import tqdm

import math
import jax

import numpy as np
import jax.numpy as jnp
from jax import random
import tensorflow_probability as tfp
from tensorflow_probability.substrates.jax import distributions
import jax.nn as nn


import optax

from collections import namedtuple
from functools import partial
from jax import grad, random, vmap, jit
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree

def init_params(rng_key, args):
    dim = args.N**2
    sizes = [dim, 256, 256, 256, 3*dim]
    rng_key, rng_key_mlp = random.split(rng_key)
    wnb = mlp_init_params(sizes, rng_key_mlp) 
    params = {"wnb": wnb}
    if args.cv == "lrn":
        params["cv"] = jnp.array(0.)
    return rng_key, params

def model(params, x):
    # Returns flip_logits
    dim = x.shape[-1]
    logits = mlp_forward(params["wnb"], x, activation_fn=jax.nn.swish)
    # return jax.vmap(partial(mlp_forward, params, activation_fn=jax.nn.swish))(x)
    add_val_logits, del_loc_logits = logits[:2 * dim], logits[2 * dim:]
    return add_val_logits.reshape(-1, 2), del_loc_logits 

def state_mask(state, init_zero, mask_null=False):
    # Return mask containing 1 at non-null locations
    if init_zero:
        if mask_null:
            mask = jnp.where(state.abs() < 1e-8, 1., 0.)
        else:
            mask = jnp.where(state != 0, 1., 0.)
    else:
        if mask_null:
            mask = jnp.where(state < -0.5, 1., 0.)
        else:
            mask = jnp.where(state > -0.5, 1., 0.)
    return mask

def index_to_value(index, init_zero):
    return 2 * index - 1 if init_zero else index

def value_to_index(value, init_zero):
    return jnp.floor_divide(value + 1, 2) if init_zero else value

@partial(jax.jit, static_argnums=(2, 3))
def sample_forward(rng_key, gfn_params, dim, init_zero, trajectory=None):
    state = -jnp.ones((dim,))     
    if init_zero: 
        state = state * 0 
    log_pf, log_pb = jnp.zeros((dim+1)), jnp.zeros((dim+1))

    for step in range(1, dim+1):
        mask = state_mask(state, init_zero)
        add_val_logits, _ = model(gfn_params, state)

        if trajectory is not None:
            add_loc = (trajectory[step-1] != trajectory[step]).nonzero(size=1, fill_value=-1)[0].squeeze() # statically typed with size=1
            assert add_loc != -1, "Consecutive states need to differ by exactly on bit!"
            add_vidx_logprob = jax.nn.log_softmax(add_val_logits[add_loc])
            add_val = trajectory[step, add_loc]
            add_vidx = value_to_index(add_val, init_zero)
        else:
            rng_key, rng_key_loc, rng_key_vidx = random.split(rng_key, 3)
            add_loc_logits = jnp.log((1 - mask) / (1e-9 + (1 - mask).sum(0, keepdims=True))) # uniform prob over all null locations
            add_loc = random.categorical(rng_key_loc, logits=add_loc_logits) 
            add_vidx_logprob = jax.nn.log_softmax(add_val_logits[add_loc])
            add_vidx = random.categorical(rng_key_vidx, logits=add_vidx_logprob)
            add_val = index_to_value(add_vidx, init_zero)
        state = state.at[add_loc].set(add_val)
        log_pf = log_pf.at[step].set(add_vidx_logprob[add_vidx] - jnp.log(dim - step + 1))
        log_pb = log_pb.at[step].set(-jnp.log(step))
    return state, log_pf, log_pb

@partial(jax.jit, static_argnums=(1, 3, 4))
def vmapped_sample_forward(rng_key, batch_size, gfn_params, dim, init_zero):
    rng_keys = random.split(rng_key, batch_size)
    return jax.vmap(sample_forward, in_axes=[0, None, None, None])(rng_keys, gfn_params, dim, init_zero)

def log_p_fwd(rng_key, gfn_params, dim, init_zero):
    samples_fwd, log_pf_fwd, log_pb_fwd = sample_forward(rng_key, gfn_params, dim, init_zero)
    log_pf_fwd = log_pf_fwd.sum(axis=-1)
    log_pb_fwd = log_pb_fwd.sum(axis=-1)
    aux = (samples_fwd, log_pb_fwd)
    return log_pf_fwd, aux 

def valgrad_log_p_fwd(rng_key, gfn_params, dim, init_zero):
    val, grad = jax.value_and_grad(log_p_fwd, argnums=1, has_aux=True)(rng_key, gfn_params, dim, init_zero)
    return val, grad

@partial(jax.jit, static_argnums=(1,    3, 4))
def per_sample_valgrad_log_p_fwd(rng_key, batch_size, gfn_params, dim, init_zero):
    rng_keys = random.split(rng_key, batch_size)
    vals, grads = jax.vmap(valgrad_log_p_fwd, in_axes=[0, None, None, None])(rng_keys, gfn_params, dim, init_zero)
    return vals, grads

def sample_backward(dim, batch_size, init_zero, data):
    batch_size = data.shape[0]
    trajectory, _, log_pf, log_pb = empty_trajectory(batch_size, data=data)
    state = data

    for step in range(dim+1, 0, -1):
        del_val_logits, _ = get_logits(state)

        if step < dim + 1: # skip first (last) step
            del_val_logits = del_val_logits.reshape(-1, dim, 2)
            log_del_val_prob = del_val_logits.gather(1, del_locs.unsqueeze(2).repeat(1, 1, 2)).squeeze().log_softmax(1)
            log_pf[:, step] = log_del_val_prob.gather(1, deleted_val).squeeze(1) - math.log(dim - step + 1)
            log_pb[:, step] = -math.log(step)
            trajectory[:, step-1] = state

        if step > 0: # skip last (first) step
            mask = state_mask(state, mask_null=True)
            del_locs = (0 - 1e9 * mask).softmax(1).multinomial(1)  # row sum not need to be 1
            deleted_val= state.gather(1, del_locs).long()
            del_values = jnp.ones(batch_size, 1) * (0 if init_zero else -1)
            state = state.scatter(1, del_locs, del_values)
    return trajectory, log_pf, log_pb

def _right_expand_as(x, y):
    return x[(...,) + (None,)*(len(y.shape)-1)]

def eval0(e):
    return e - jax.lax.stop_gradient(e)

def eval1(e):
    return jnp.exp(eval0(e))

def loo_mean(v):
    n = v.shape[0]
    return (n * jnp.mean(v, axis=0, keepdims=True) - v)/(n-1)

def log_Z(lw):
    n = lw.shape[0]
    return jax.nn.logsumexp(lw, axis=0) - jnp.log(n)

def loo_log_Z(lw):
    n = lw.shape[0]
    loo_lZ = jax.nn.logsumexp(lw[loo_idx(n)], axis=1) - jnp.log(n-1)
    return loo_lZ

def loo_E_log_w(lw):
    loo_E_lw = loo_mean(lw)
    return loo_E_lw

def grad_loss_fn(rng_key, gfn, ebm, gfn_params, ebm_params, dim, back_ratio, batch_size, init_zero, cv_name):
            rkl_loss, fkl_loss, mle_loss = jnp.array(0.), jnp.array(0.), jnp.array(0.)
            # rng_key_fwd, rng_key_bwd = random.split(rng_key, 2)

            if back_ratio < 1.:
                vals, h_dict = per_sample_valgrad_log_p_fwd(rng_key, batch_size, gfn_params, dim, init_zero)
                log_pf_fwd, aux = vals
                samples_fwd, log_pb_fwd = aux
                score_value_fwd = ebm.vmapped_model(ebm_params, samples_fwd) 
                log_w = log_pb_fwd + score_value_fwd - log_pf_fwd
                f = -log_w

                g = jax.tree_map(lambda h: jnp.mean(_right_expand_as(f, h)*h, axis=0), h_dict)
                rkl_loss = jnp.mean(f)
                if cv_name == "log_Z":
                    c_log_Z = loo_mean(log_w)
                    g_prime = jax.tree_map(lambda h: jnp.mean(_right_expand_as(f + c_log_Z, h)*h, axis=0), h_dict)
                elif cv_name == "E_log_w":
                    c_E_log_w = loo_E_log_w(log_w)
                    g_prime = jax.tree_map(lambda h: jnp.mean(_right_expand_as(f + c_E_log_w, h)*h, axis=0), h_dict)
                elif cv_name == "opt":
                    var_h = jax.tree_util.tree_reduce(lambda s, h: s + jnp.sum(jnp.mean(h**2, axis=0)), h_dict, initializer=0.)
                    cov_gh = jax.tree_util.tree_reduce(lambda s, h: s + jnp.sum(jnp.mean(_right_expand_as(f, h)*h**2, axis=0)), h_dict, initializer=0.)
                    c_opt = - cov_gh / var_h
                    g_prime = jax.tree_map(lambda h: jnp.mean(_right_expand_as((f + c_opt), h)*h, axis=0), h_dict)
                elif cv_name == "opt_vec":
                    var_d = jax.tree_map(lambda h: jnp.mean(h**2, axis=0), h_dict)
                    cov_d = jax.tree_map(lambda h: jnp.mean(_right_expand_as(f, h)*h**2, axis=0), h_dict)
                    c_opt = jax.tree_map(lambda var_d, cov_d: jnp.nan_to_num(- cov_d / var_d, nan=1.0), var_d, cov_d)
                    g_prime = jax.tree_map(lambda h, c: jnp.mean((_right_expand_as(f, h) + c) *h, axis=0), h_dict, c_opt)
                elif cv_name == "none":
                    g_prime = g
                else:
                    raise NotImplementedError

            if back_ratio > 0.:
                raise NotImplementedError

            wkl_loss = (1 - back_ratio) * rkl_loss + back_ratio * fkl_loss
            return wkl_loss, g_prime, g

def loss_fn(rng_key, gfn, ebm, gfn_params, ebm_params, dim, back_ratio, batch_size, init_zero, cv_name):
            rkl_loss, fkl_loss, mle_loss = jnp.array(0.), jnp.array(0.), jnp.array(0.)
            # rng_key_fwd, rng_key_bwd = random.split(rng_key, 2)
            if back_ratio < 1.:
                samples_fwd, log_pf_fwd, log_pb_fwd = vmapped_sample_forward(rng_key, batch_size, gfn_params, dim, init_zero)
                score_value_fwd = ebm.vmapped_model(ebm_params, samples_fwd) 
                log_pf_fwd = log_pf_fwd.sum(axis=-1)
                log_pb_fwd = log_pb_fwd.sum(axis=-1)

                log_w = log_pb_fwd + score_value_fwd - log_pf_fwd
                if cv_name == "lrn":
                    cv = gfn_params['cv']
                elif cv_name == "none":
                    cv = 0.
                else:
                    raise NotImplementedError

                f = -log_w + cv
                rkl_loss = eval1(f) * jax.lax.stop_gradient(f) # --grad-> 1 * 0 + gf * f
                return rkl_loss.mean()

            if back_ratio > 0.:
                raise NotImplementedError

            wkl_loss = (1 - back_ratio) * rkl_loss + back_ratio * fkl_loss
            return wkl_loss

def mlp_random_params(shape, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, shape), scale * random.normal(b_key, shape[0:1])

def mlp_init_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [mlp_random_params((n, m), k) for m, n, k, in zip(sizes[:-1], sizes[1:], keys)]

def mlp_forward(params, x, activation_fn):
    activations = x.T 
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = activation_fn(outputs)
    final_w, final_b = params[-1]
    outputs = jnp.dot(final_w, activations) + final_b
    return outputs

if __name__ == "__main__":
    mlp_layer_sizes = [784, 784, 784, 784]
    step_size = 0.01
    num_epochs = 10
    batch_size = 128
    n_targets = 10

    params = mlp_init_params(mlp_layer_sizes, random.PRNGKey(0))
    x = random.normal(random.PRNGKey(1), (28*28,))
    y = mlp_forward(params, x, nn.relu)
