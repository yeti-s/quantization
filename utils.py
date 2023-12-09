import torch
import numpy as np
import matplotlib.pyplot as plt


def draw_tensor_3D(tensor, x_label="X", y_label="Y", scale=5):
    data = tensor.numpy()

    if data.shape[0] == 1:
        data = data.squeeze(0)
    
    shape = data.shape
    x = np.arange(0, shape[1], 1)
    y = np.arange(0, shape[0], 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, data, cmap='viridis')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel('value')

    max = np.max(data)
    if scale < max:
        scale = max
    ax.set_zlim(0, scale)

    plt.show()

def draw_activation(tensor:torch.Tensor, scale:int=5):
    tensor = tensor.detach().abs().to('cpu')
    tensor = torch.abs(tensor)
    draw_tensor_3D(tensor, "Channel", "Token", scale)

def draw_weight(tensor:torch.Tensor, scale:int=5):
    tensor = tensor.detach().abs().to('cpu')
    tensor = torch.abs(tensor)
    draw_tensor_3D(tensor, "Out Channel", "In Channel", scale)