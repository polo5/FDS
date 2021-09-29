"""
This is to check that Theorem 4.1 holds in the case where all the cross term of the covariance matrix are zero, i.e.
each hypergradient is independant of all other hypergradients. We also use a constant variance=sigma^2 for all steps
"""


import numpy as np
import random
from utils.helpers import *

class ProofChecker(object):
    def __init__(self, args):
        self.args = args
        self.args.T = args.T - args.T%args.W # make sure we have a whole number of windows that fit inside horizon
        self.K = int(args.T/args.W)
        print(f'Running experiments for a total of T={self.args.T} while using {self.K} hyperparameters, each shared over W={args.W} contiguous steps')
        print(f'not sharing: expected MSE = sigma^2 = {args.sigma**2}')
        print(f'sharing: expected MSE for min drift = sigma^2/W = {args.sigma**2/args.W}')
        print(f'sharing: expected MSE for max drift (upper bound) = sigma^2/W + eps^2(W^2-1)/12 = {args.sigma**2/args.W + args.epsilon**2*(args.W**2 - 1)/12}')

    def sample_min_drift(self):
        """
        epsilon_t = 0 for all time steps
        """
        hypergrad_means = np.array([self.args.mu_0 for _ in range(self.args.T)])
        hypergrads = np.random.normal(hypergrad_means, self.args.sigma, size=(self.args.n_seeds, self.args.T))
        optimal_hypergrads = hypergrad_means
        return hypergrads, optimal_hypergrads

    def sample_max_drift(self):
        """
        epsilon_t = epsilon for all time steps
        """
        hypergrad_means = np.array([self.args.mu_0 + n*self.args.epsilon for n in range(self.args.T)])
        hypergrads = np.random.normal(hypergrad_means, self.args.sigma, size=(self.args.n_seeds, self.args.T))
        optimal_hypergrads = hypergrad_means
        return hypergrads, optimal_hypergrads


    def sample_random_drift(self):
        epsilons = np.random.uniform(-self.args.epsilon, self.args.epsilon, self.args.T-1)
        hypergrad_means = [self.args.mu_0]
        for eps in epsilons:
            hypergrad_means.append(hypergrad_means[-1]+eps)
        hypergrad_means = np.array(hypergrad_means)
        hypergrads = np.random.normal(hypergrad_means, self.args.sigma, size=(self.args.n_seeds, self.args.T))
        optimal_hypergrads = hypergrad_means
        return hypergrads, optimal_hypergrads

    def mse_not_sharing(self, hypergrads, optimal_hypergrads):
        return np.mean((hypergrads - optimal_hypergrads)**2)

    def mse_sharing(self, hypergrads, optimal_hypergrads):
        hypergrads_after_sharing = [np.mean(h.reshape((self.K, self.args.W)), axis=1).repeat(self.args.W) for h in hypergrads]
        hypergrads_after_sharing = np.array(hypergrads_after_sharing)
        return np.mean((hypergrads_after_sharing - optimal_hypergrads)**2)

    def run(self):
        print('\nMIN DRIFT:')
        hypergrads, optimal_hypergrads = self.sample_min_drift()
        mse_not_sharing, mse_sharing = self.mse_not_sharing(hypergrads, optimal_hypergrads), self.mse_sharing(hypergrads, optimal_hypergrads)
        print(f'actual mse when not sharing = {mse_not_sharing:.5f} --- mse sharing = {mse_sharing:.5f}')

        print('\nMAX DRIFT:')
        hypergrads, optimal_hypergrads = self.sample_max_drift()
        mse_not_sharing, mse_sharing = self.mse_not_sharing(hypergrads, optimal_hypergrads), self.mse_sharing(hypergrads, optimal_hypergrads)
        print(f'actual mse not sharing = {mse_not_sharing:.5f} --- mse sharing = {mse_sharing:.5f}')

        print('\nRANDOM DRIFT:')
        hypergrads, optimal_hypergrads = self.sample_random_drift()
        mse_not_sharing, mse_sharing = self.mse_not_sharing(hypergrads, optimal_hypergrads), self.mse_sharing(hypergrads, optimal_hypergrads)
        print(f'actual mse not sharing = {mse_not_sharing:.5f} --- mse sharing = {mse_sharing:.5f}')


def main(args):
    np.random.seed(args.seed)

    t0 = time.time()
    proof = ProofChecker(args)
    proof.run()
    print(f'\nTotal time: {format_time(time.time() - t0)}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Welcome to GreedyGrad')
    parser.add_argument('--T', type=int, default=400, help='total number of inner_steps or batches, where each batch would make use of a different hyperparameter')
    parser.add_argument('--n_seeds', type=int, default=10000, help='how many seeds to sample. Each seed = T hypergradients')
    parser.add_argument('--W', type=int, default=40, help='window to share hyperparameters from contiguous steps over')
    parser.add_argument('--mu_0', type=float, default=0.0)
    parser.add_argument('--sigma', type=float, default=0.25)
    parser.add_argument('--epsilon', type=float, default=0.08)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()

    assert args.W%2==0, "even W required for lower bound of MSE_shared to be right"

    main(args)



    ### NOTES:
    # sharing should always help in min_drift/random_drift setting, but needs small W to help in max_drift setting




