import matplotlib.pyplot as plt
import torch
import numpy as np
import glob
import imageio
import os


@torch.no_grad()
def plot_gradients(model, data, epoch):
    range = 4
    xx = np.stack(np.meshgrid(np.linspace(-range, range, 50),
                              np.linspace(-range, range, 50)), axis=-1).reshape(-1, 2)
    xx = torch.tensor(xx).float()
    scores = model(xx)
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores / (scores_norm + 1e-9) * np.log1p(scores_norm)
    # Perform the plots
    plt.figure(figsize=(10, 10))
    plt.scatter(*data.t().numpy(),  s=0.5)
    plt.quiver(*xx.T, *scores_log1p.T, width=0.002)
    plt.xlim(-range, range)
    plt.ylim(-range, range)
    plt.savefig(f'images/epoch_{epoch}.png')
    plt.close()


def make_a_gif(images='images/*.png', path='gradients.gif'):
    filenames = glob.glob(images)
    filenames.sort(key=lambda x: int(x.split('/')[1][6:-4]))
    images = [imageio.imread(filename) for filename in filenames]

    imageio.mimsave(path, images, duration=0.2)
    os.system(f'gifsicle --scale 0.7 -O3 {path} -o {path} ')
    print('Done.')
