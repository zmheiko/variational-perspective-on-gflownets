from functools import partial
from tqdm import tqdm
from numpy.random import rand
import math
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

def init_params(rng_key, args):
    return rng_key, {"n": args.N, "beta": args.beta}

# @partial(jax.jit, static_argnums=(0,))
def square_lattice_adjacency_matrix(n):
    if not n > 2:
        raise Exception("can only be used for n > 2")
    row = [0, 1] + (n-3)*[0] + [1]
    offdi = la.circulant(row)
    I = jnp.eye(n)
    A = jnp.kron(offdi,I) + jnp.kron(I,offdi)
    return A

@partial(jax.jit, static_argnums=(0,1))
def model(n, beta, x):
    J = beta * square_lattice_adjacency_matrix(n)
    H = -jnp.dot(x, jnp.dot(J, x))/2. # compensate for overcounting by factor 2
    return -H # log energy

def vmapped_model(params, x):
    return jax.vmap(model, in_axes=[None, None, 0])(params["n"], params["beta"], x) 

class IsingModel():
    def __init__(self, n, beta=1, device=None):
        self.n = n
        self.d = n**2
        self.beta = beta
        self.A = square_lattice_adjacency_matrix(self.n)
        self.J = self.beta * self.A
        
    def __call__(self, *args, **kwargs):
        return self.simulate_mh(*args, **kwargs)

    def log_energy(self, s):
        return -self.hamiltonian(s)

    def hamiltonian(self, s):
        return -jnp.einsum("bi,ij,bj->b", s, self.J, s)/2. # compensate for overcounting by factor 2

    def simulate_mh(self, steps, num_chains=1, return_chains=False, debug=False, print_progress=False):
        s = 2*np.random.randint(2, size=(num_chains, self.n, self.n)) - 1.

        ar = np.zeros(3)
        if return_chains:
            state_hist = np.zeros((num_chains, steps, self.n, self.n))
            state_hist[:, 0, :] = s 

        pbar = range(steps)
        pbar = tqdm(pbar) if print_progress else pbar
        for t in pbar:
            a = np.random.randint(0, self.n)
            b = np.random.randint(0, self.n)
            s_pos = s[..., a, b]
            nb = s[..., (a+1)%self.n, b] + s[..., a, (b+1)%self.n] + s[..., (a-1)%self.n, b] + s[..., a, (b-1)%self.n]
            ##
            if debug:
                s_prop = s.copy()
                s_prop[..., a, b] = np.negative(s[..., a, b])
                H = self.hamiltonian(s.reshape(s.shape[0], self.n**2))
                H_prop = self.hamiltonian(s_prop.reshape(s_prop.shape[0], self.n**2))
                H_diff = self.beta * 2*s_pos*nb
                assert np.isclose(H_diff, H_prop-H).all()
            cost = 2*s_pos*nb
            s_pos = np.where(np.random.rand(num_chains) < np.exp(-cost*self.beta), -s_pos, s_pos)
            s[..., a, b] = s_pos
            if return_chains:
                state_hist[:, t, :] = s 

        if return_chains:
            return state_hist, ar
        return s, None

def plot_grid(grids, num_cols=4, *args, **kwargs):
    n = len(grids)
    if n < num_cols:
        num_cols = n
    num_rows = (n - 1)// num_cols + 1
    fig = plt.figure(figsize=np.array([num_cols, num_rows])*2)
    gs = gridspec.GridSpec(num_rows, num_cols)
    for i in range(n):
        ax = plt.subplot(gs[i // num_cols, i % num_cols])
        ax.matshow(grids[i], *args, **kwargs)
    return fig
