# Jacobian and score matching loss is adapted from https://github.com/acids-ircam/diffusion_models

import torch
import torch.nn as nn
import torch.autograd as autograd


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    # Build an MLP network.
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def jacobian(f, x):
    """
    Computes the Jacobian
    """
    B, N = x.shape
    y = f(x)
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = autograd.grad(y, x, grad_outputs=v, retain_graph=True,
                                create_graph=True, allow_unused=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=2).requires_grad_()
    return jacobian


def score_matching(model, samples):
    samples.requires_grad_(True)
    logp = model(samples)
    norm_jacobian = torch.norm(logp, dim=-1) ** 2 / 2.
    jacob_mat = jacobian(model, samples)
    trace_jacobian = torch.diagonal(jacob_mat, dim1=-2, dim2=-1).sum(-1)
    loss = (trace_jacobian + norm_jacobian).mean(-1)
    return loss


