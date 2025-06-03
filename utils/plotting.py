import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

def plot_lorenz_predictions(x_pred, x_true=None, fig=None):

    if isinstance(x_pred, torch.Tensor):
        x_pred = x_pred.detach().cpu().numpy()
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().cpu().numpy()

    if fig is None:
        fig = plt.figure(figsize=(11,6))

    if x_true is not None:
        gs = GridSpec(nrows=3, ncols=2, width_ratios=(0.6, 0.4), figure=fig, wspace=0.3) 
        ax1 = fig.add_subplot(gs[:,0], projection='3d')
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1], sharex=ax2)
        ax4 = fig.add_subplot(gs[2,1], sharex=ax2)

        ax2.tick_params('x', labelbottom=False)
        ax3.tick_params('x', labelbottom=False)
        ax3.tick_params('y', labelleft=False)

        axes = [ax2, ax3, ax4]
    else:
        ax1 = fig.add_subplot(projection='3d')

    ax1.plot(*x_pred.T, lw=0.2, color='r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    if x_true is not None:

        max_ind = min(x_true.shape[0], x_pred.shape[0])
        var_names = ['x','y','z']

        for i, ax in enumerate(axes):
            ax.plot(x_true[:max_ind, i], color='k', linewidth=1, label='True')
            ax.plot(x_pred[:max_ind, i], color='r', linewidth=1, label='Prediction')
            ax.set_ylabel(var_names[i])
            ax.grid(True, linestyle=':', alpha=0.5)

        ax4.set_xlabel('Time')


# --------------------------------------------------------------------- #
def plot_celegans_neural_data(data, fig=None):

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    T, C = data.shape

    if fig is None:
        fig_height = min(25, C * 1.5)
        fig = plt.figure(figsize=(10, fig_height))

    channel_separation = np.mean(np.percentile(data, 90, axis=0) - np.percentile(data, 10, axis=0))


    for i in range(C):
        plt.plot(data[:,i] + 5 * channel_separation * i, color='k', linewidth=1)

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xlabel('Time step')
    plt.yticks([5 * channel_separation * i for i in range(C)], [f'Channel {i}' for i in range(C)])

# --------------------------------------------------------------------- #
def plot_celegans_neural_predictions(x_pred, x_true=None, fig=None):

    if isinstance(x_pred, torch.Tensor):
        x_pred = x_pred.detach().cpu().numpy()
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().cpu().numpy()

    if fig is None:
        fig = plt.figure(figsize=(11,6))

    channels = x_pred.shape[1]
    rows = min(5, channels)

    gs = GridSpec(nrows=rows, ncols=2, width_ratios=(0.6, 0.4), figure=fig, wspace=0.3) 
    axes = []
    for i in range(rows):
        ax = fig.add_subplot(gs[i,1])
        axes.append(ax)

    ax_3d = fig.add_subplot(gs[:,0], projection='3d')

    # Plot up to 5 timeseries on the right
    max_ind = min(x_true.shape[0], x_pred.shape[0])
    for i, ax in enumerate(axes):
        ax.plot(x_true[:max_ind, i], color='k', linewidth=1, label='True')
        ax.plot(x_pred[:max_ind, i], color='r', linewidth=1, label='Prediction')
        ax.grid(True, linestyle=':', alpha=0.5)

    
    # Plot the predicted trajectory in 3d (if more than 3 features, do a PCA)
    if channels > 3:
        x_pred = PCA(n_components=3).fit_transform(x_pred)
        
    ax_3d.plot(*x_pred.T, lw=0.2, color='r')
    ax_3d.set_xlabel('x')
    ax_3d.set_ylabel('y')
    ax_3d.set_zlabel('z')
