from util import sample_2d_data
import torch
from model import mlp, score_matching
from vis_utils import plot_gradients, make_a_gif

device = 'cpu'
dataset = sample_2d_data(dataset='8gaussians', n_samples=10000)
model = mlp(sizes=[2, 128, 128, 2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_and_visualize():
    for epoch in range(100):
        loss = score_matching(model, dataset)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(epoch)
        plot_gradients(model, dataset, epoch)


if __name__ == '__main__':
    train_and_visualize()
    make_a_gif(images=f'images/*.png', path='gradient.gif')
