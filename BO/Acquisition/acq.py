import math
from numbers import Real

import numpy as np
import torch
from platypus import Problem, Solution, InjectedPopulation, Archive, NSGAII
from scipy.stats import norm
import botorch.acquisition
import torch.nn as nn

"""
class acq:
    input:(那一个acq，batch)
    ucb：
    ei：f_best
    pi：
def way：
    
    output：（返回候选点）
"""


class UCB:
    def __init__(self, mean_func, variance_func, kappa=2.0):
        """
        Initialize the Batch Upper Confidence Bound (UCB) acquisition function.

        Args:
            mean_func (callable): Function to compute the mean of the GP at given points (PyTorch tensor).
            variance_func (callable): Function to compute the variance of the GP at given points (PyTorch tensor).
            kappa (float): Controls the balance between exploration and exploitation.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.kappa = kappa
    # forward
    def compute_ucb(self, X):
        """
        Compute the UCB values for the given inputs.

        Args:
            X (torch.Tensor): The input points where UCB is to be evaluated.

        Returns:
            torch.Tensor: The UCB values for the input points.
        """
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        print(mean)
        print(variance)
        return mean + self.kappa * torch.sqrt(variance)

    def find_next_batch(self, bounds, batch_size=1, n_samples=1000):
        """
        Find the next batch of points to sample by selecting the ones with the highest UCB from a large set of random samples.

        Args:
            bounds (np.ndarray): The bounds for each dimension of the input space.
            batch_size (int): The number of points in the batch.
            n_samples (int): The number of random points to sample for finding the maximum UCB.

        Returns:
            torch.Tensor: The next batch of points to sample.
        """
        X_selected = []
        for _ in range(batch_size):
            # Generate a large number of random points
            X_random = torch.FloatTensor(n_samples, bounds.shape[0]).uniform_(bounds[0, 0], bounds[0, 1])

            # Compute UCB for all random points
            UCB_values = self.compute_ucb(X_random)

            # Select the point with the highest UCB value
            idx_max = torch.argmax(UCB_values)
            X_selected.append(X_random[idx_max])
        return torch.stack(X_selected)


def EI(input_mean_function, input_std_function, x, f_best, xi=0.01):
    """
    计算期望提升 (Expected Improvement, EI)

    Args:
        input_mean_function (callable): 预测均值的函数，返回矩阵。
        input_std_function (callable): 预测标准差的函数，返回矩阵。
        x (float or torch.Tensor): 输入数据点。
        f_best (float): 到目前为止观察到的最佳函数值。
        xi (float): 探索参数。

    Returns:
        torch.Tensor: EI 分数矩阵。
    """
    mean_matrix = input_mean_function(x)
    std_matrix = input_std_function(x)

    # 防止标准差为零
    std_matrix = torch.clamp(std_matrix, min=1e-9)

    Z = (mean_matrix - f_best - xi) / std_matrix
    ei_matrix = (mean_matrix - f_best - xi) * torch.Tensor(norm.cdf(Z.numpy())) + std_matrix * torch.Tensor(norm.pdf(Z.numpy()))
    return ei_matrix

class EI:
    def __init__(self, mean_func, variance_func, xi=0.01):
        """
        Initialize the Expected Improvement (EI) acquisition function.

        Args:
            mean_func (callable): Function to compute the mean of the GP at given points.
            variance_func (callable): Function to compute the variance of the GP at given points.
            xi (float): Controls the balance between exploration and exploitation.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.xi = xi

    def compute_ei(self, X, f_best):
        """
        Compute the EI values for the given inputs.

        Args:
            X (torch.Tensor): The input points where EI is to be evaluated.
            f_best (float): The best observed objective function value to date.

        Returns:
            torch.Tensor: The EI values for the input points.
        """
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - f_best - self.xi) / std
        ei = (mean - f_best - self.xi) * torch.tensor(norm.cdf(Z.numpy()), dtype=torch.float32) + std * torch.tensor(norm.pdf(Z.numpy()), dtype=torch.float32)
        return ei

    def find_next_batch(self, bounds, batch_size=1, n_samples=1000, f_best=None):
        """
        Find the next batch of points to sample by selecting the ones with the highest EI from a large set of random samples.

        Args:
            bounds (np.ndarray): The bounds for each dimension of the input space.
            batch_size (int): The number of points in the batch.
            n_samples (int): The number of random points to sample for finding the maximum EI.
            f_best (float): The best observed objective function value to date.

        Returns:
            torch.Tensor: The next batch of points to sample.
        """
        if f_best is None:
            raise ValueError("f_best must be provided for EI calculation.")

        X_selected = []
        for _ in range(batch_size):
            # Generate a large number of random points
            X_random = torch.FloatTensor(n_samples, bounds.shape[0]).uniform_(bounds[0, 0], bounds[0, 1])

            # Compute EI for all random points
            EI_values = self.compute_ei(X_random, f_best)

            # Select the point with the highest EI value
            idx_max = torch.argmax(EI_values)
            X_selected.append(X_random[idx_max])
        return torch.stack(X_selected)

class PI:
    def __init__(self, mean_func, variance_func, sita=0.01):
        """
        Initialize the Probability of Improvement (PI) acquisition function.

        Args:
            mean_func (callable): Function to compute the mean of the GP at given points.
            variance_func (callable): Function to compute the variance of the GP at given points.
            xi (float): Controls the balance between exploration and exploitation.
        """
        self.mean_func = mean_func
        self.variance_func = variance_func
        self.sita = sita

    def compute_pi(self, X, f_best):
        """
        Compute the PI values for the given inputs.

        Args:
            X (torch.Tensor): The input points where PI is to be evaluated.
            f_best (float): The best observed objective function value to date.

        Returns:
            torch.Tensor: The PI values for the input points.
        """
        mean = self.mean_func(X)
        variance = self.variance_func(X)
        std = torch.sqrt(variance)

        # Preventing division by zero in standard deviation
        std = torch.clamp(std, min=1e-9)

        Z = (mean - f_best - self.sita) / std
        pi = torch.tensor(norm.cdf(Z.numpy()), dtype=torch.float32)
        return pi

    def find_next_batch(self, bounds, batch_size=1, n_samples=1000, f_best=None):
        """
        Find the next batch of points to sample by selecting the ones with the highest PI from a large set of random samples.

        Args:
            bounds (np.ndarray): The bounds for each dimension of the input space.
            batch_size (int): The number of points in the batch.
            n_samples (int): The number of random points to sample for finding the maximum PI.
            f_best (float): The best observed objective function value to date.

        Returns:
            torch.Tensor: The next batch of points to sample.
        """
        if f_best is None:
            raise ValueError("f_best must be provided for PI calculation.")

        X_selected = []
        for _ in range(batch_size):
            # Generate a large number of random points
            X_random = torch.FloatTensor(n_samples, bounds.shape[0]).uniform_(bounds[0, 0], bounds[0, 1])

            # Compute PI for all random points
            PI_values = self.compute_pi(X_random, f_best)

            # Select the point with the highest PI value
            idx_max = torch.argmax(PI_values)
            X_selected.append(X_random[idx_max])
        return torch.stack(X_selected)

def find_candidate_point(acquisition, bounds, batch_size=1, n_samples=1000):
    """
    Find the next batch of points to sample by selecting the ones with the highest UCB from a large set of random samples.

    Args:
        compute_ucb (callable): Function to compute the UCB values.
        bounds (np.ndarray): The bounds for each dimension of the input space.
        batch_size (int): The number of points in the batch.
        n_samples (int): The number of random points to sample for finding the maximum UCB.

    Returns:
        torch.Tensor: The next batch of points to sample.
    """
    X_selected = []
    for _ in range(batch_size):
        # Generate a large number of random points
        X_random = torch.FloatTensor(n_samples, bounds.shape[0]).uniform_(bounds[0, 0], bounds[0, 1])

        # Compute UCB for all random points
        UCB_values = acquisition(X_random)

        # Select the point with the highest UCB value
        idx_max = torch.argmax(UCB_values)
        X_selected.append(X_random[idx_max])
    return torch.stack(X_selected)
