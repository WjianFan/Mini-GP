from cigp_test import cigp
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from MiniGP.kernel import ARDKernel
from BO.Acquisition.acq import UCB, EI, PI

# Define the objective function

import gpytorch
import numpy as np
import matplotlib.pyplot as plt

# Define a 5-dimensional objective function (for demonstration purposes)
def objective_function(x):
    # Simple sum of sine functions for demonstration
    return torch.sin(x)+torch.sin(3*x)


# Initialize prior knowledge with 10 random points
input_dim = 1
num_initial_points = 5
train_x = torch.rand(num_initial_points, input_dim) * 10  # Random points in [0, 10] for each dimension
train_y = objective_function(train_x).reshape(-1,1)

# Define a simple exact GP model using GPyTorch
model = cigp(train_x,train_y)

# Define the mean and variance functions for UCB
def mean_function(X):
    model.eval()
    with torch.no_grad():
        mean, _ = model.forward(X)
        return mean

def variance_function(X):
    model.eval()
    with torch.no_grad():
        _, var = model.forward(X)
        return var

# Initialize UCB
ucb = UCB(mean_function, variance_function, kappa=5)
pi = PI(mean_function, variance_function)
ei = EI(mean_function, variance_function)

best_y = []
# use it to remember the key iteration
key_iterations = [2,4,5,6,8,10]
predictions = []
iteration_label = True

# Bayesian optimization loop
bounds = np.array([[0, 10]] * input_dim)
for iteration in range(10):  # Run for 5 iterations
    # fit the gp model
    model = cigp(train_x, train_y)
    model.train_adam(150,0.01)
    # Find the next batch of poinuts
    batch_points = ucb.find_next_batch(bounds, batch_size=1, n_samples=1000)
    # batch_points = pi.find_next_batch(bounds, batch_size=1, n_samples=1000, f_best=train_x[np.argmax(train_y)])
    # batch_points = ei.find_next_batch(bounds, batch_size=1, n_samples=1000, f_best=train_x[np.argmax(train_y)])
    #find_next_batch(acq)
    batch_points = torch.tensor(batch_points).float()

    # Evaluate the objective function
    new_y = objective_function(batch_points.squeeze()).reshape(-1,1)

    # Update the model
    train_x = torch.cat([train_x, batch_points])
    train_y = torch.cat([train_y, new_y])
    # Store the best objective value found so far
    best_y.append(new_y.max().item())
    # Visualization

    # 在关键迭代时保存模型预测
    if (iteration + 1) in key_iterations:
        model.eval()
        if iteration_label == True:
            fixed_dims = torch.full((1, input_dim - 1), 5.0)  # Example: set them to the midpoint (5.0)
            test_points = torch.linspace(0, 10, 100)
            test_X = torch.cat((test_points.unsqueeze(1), fixed_dims.expand(test_points.size(0), -1)), 1)
            iteration_label = False
        true_y = objective_function(test_X)

        with torch.no_grad():
            pred_mean, pred_std = model.forward(test_X)
            predictions.append((pred_mean, pred_std))


# 绘制子图
plt.figure(figsize=(10, 8))
for i, (pred_mean, pred_std) in enumerate(predictions):
    plt.subplot(3, 2, i+1)
    plt.ylim(-5, 5)
    plt.plot(test_points.numpy(), true_y.numpy(), 'k-', label='True function')
    plt.plot(test_points.numpy(), pred_mean.numpy(), 'b--', label='GP mean')
    plt.fill_between(test_points.numpy().reshape(-1),
                     (pred_mean - 20 * pred_std).numpy().reshape(-1),
                     (pred_mean + 20 * pred_std).numpy().reshape(-1),
                     color='blue', alpha=0.2, label='GP uncertainty')

    observed_x = train_x[:, 0].numpy()  # Only the first dimension for all observed points
    observed_y = train_y.numpy()
    plt.scatter(observed_x[:num_initial_points+key_iterations[i]], observed_y[:num_initial_points+key_iterations[i]], c='r', zorder=3, label='Observed points')
    plt.title(f'Samples: {key_iterations[i]}')
    plt.legend()

plt.tight_layout()
plt.show()