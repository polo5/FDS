"""
This is to check that Theorem 4.1 holds in the case where each step has its own variance,
and where all steps are correlated with one another
"""


import numpy as np
import random
from sklearn.datasets import make_spd_matrix
from utils.helpers import *

class ProofChecker(object):
    def __init__(self, args):
        self.args = args
        self.args.T = args.T - args.T%args.W # make sure we have a whole number of windows that fit inside horizon
        self.K = int(args.T/args.W)
        T, W, eps, c = args.T, args.W, args.epsilon, args.c
        # self.correlation_matrix = np.random.uniform(low=-args.c, high=args.c, size=(args.T, args.T))
        # np.fill_diagonal(self.correlation_matrix, 1)
        # self.sigmas = np.random.uniform(low=0, high=args.max_sigma, size=args.T)
        # self.covariance_matrix = np.diag(self.sigmas)@self.correlation_matrix@np.diag(self.sigmas)

        ## Correlation matrix has lots of different values, maximum is c
        # self.covariance_matrix = make_spd_matrix(T)/10 #random positive definite symmetric matrix
        # np.fill_diagonal(self.covariance_matrix, np.random.uniform(1,args.max_var, T)) #increase var = lower maximum correlation
        # vars = np.diag(self.covariance_matrix)
        # stds = np.sqrt(vars)
        # self.correlation_matrix = self.covariance_matrix / np.outer(stds, stds)
        # np.fill_diagonal(self.correlation_matrix, 0)
        # c0 = np.max(np.abs(self.correlation_matrix)) #max correlation
        # assert c0 < 1

        ## worst case correlation matrix has c for all it's non-diagonal entries
        # we still need the covariance to be positive semi definite. It can be shown that
        # if all off-diagonal entries of the TxT matrix are equal to c, then we need c >= -1/(T-1)
        self.correlation_matrix_worst_case = np.full((T,T), c)
        np.fill_diagonal(self.correlation_matrix_worst_case, 1)
        vars = np.random.uniform(1,args.max_var, T)
        stds = np.sqrt(vars)
        self.covariance_matrix_worst_case = self.correlation_matrix_worst_case * np.outer(stds, stds)
        print(f'sum of covariance matrix: {np.sum(self.covariance_matrix_worst_case)}')

        print(f'Running experiments for a total of T={self.args.T} while using {self.K} windows of W={W} steps, running {args.n_seeds} seeds')
        print(f'max off diagonal correlation is c: {c:.3f}')
        print(f'not sharing: expected MSE = {np.mean(vars)}')
        print(f'sharing: expected MSE upper bound for max drift = (1+c(W-1))/W)*(1/T)*sum(D_tt) + eps^2(W^2-1)/12 = {((1+c*(W-1))/W) * np.mean(vars) + eps**2*(W**2 - 1)/12}')
        # print(f'W* = best W when max drift = lower bound to optimal W otherwise = (6*sigma^2/esilon^2)^(1/3) = {(6*args.sigma**2/args.epsilon**2)**(1/3):.3f}')

    def sample_max_drift(self):
        """
        epsilon_t = epsilon for all time steps
        """
        hypergrad_means = np.array([self.args.mu_0 + n*self.args.epsilon for n in range(self.args.T)])
        # hypergrads = np.random.multivariate_normal(hypergrad_means, self.covariance_matrix, size=(self.args.n_seeds))
        hypergrads = np.random.multivariate_normal(hypergrad_means, self.covariance_matrix_worst_case, size=(self.args.n_seeds))
        optimal_hypergrads = hypergrad_means
        return hypergrads, optimal_hypergrads

    def sample_min_drift(self):
        """
        epsilon_t = 0 for all time steps
        """
        hypergrad_means = np.array([self.args.mu_0 for _ in range(self.args.T)])
        # hypergrads = np.random.multivariate_normal(hypergrad_means, self.covariance_matrix, size=(self.args.n_seeds))
        hypergrads = np.random.multivariate_normal(hypergrad_means, self.covariance_matrix_worst_case, size=(self.args.n_seeds))
        optimal_hypergrads = hypergrad_means
        return hypergrads, optimal_hypergrads

    def sample_random_drift(self):
        epsilons = np.random.uniform(-self.args.epsilon, self.args.epsilon, self.args.T-1)
        hypergrad_means = [self.args.mu_0]
        for eps in epsilons:
            hypergrad_means.append(hypergrad_means[-1]+eps)
        hypergrad_means = np.array(hypergrad_means)
        # hypergrads = np.random.multivariate_normal(hypergrad_means, self.covariance_matrix, size=(self.args.n_seeds))
        hypergrads = np.random.multivariate_normal(hypergrad_means, self.covariance_matrix_worst_case, size=(self.args.n_seeds))
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
    random.seed(args.seed)

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
    parser.add_argument('--c', type=float, default=0.1, help='used for all values in correlation matrix')
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--max_var', type=float, default=1.5, help='high values make max correlation smaller. Must be >1 to preserve semi-definite nature of covariance matrix')
    parser.add_argument('--mu_0', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    assert args.max_var >= 1
    assert args.c > -1/(args.T-1) #otherwise covariance won't be positive semi-definite

    main(args)



    ### NOTES:
    # sharing should always help in min_drift/random_drift setting, but needs small W to help in max_drift setting




