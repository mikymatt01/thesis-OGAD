from torchvision.utils import save_image
from datasets import load_dataset
import numpy as np
import torch

def saveTensorInImage(predictions_graphon, output_dir):
    predictions_graphon = torch.from_numpy(predictions_graphon).unsqueeze(0)
    save_image(predictions_graphon, output_dir)
   

def synthesize_graphon(r: int = 1000, type_idx: int = 0, alpha: int = 1) -> np.ndarray:
    """
    Synthesize graphons of Table 1
    :param r: the resolution of discretized graphon
    :param type_idx: the type of graphon
    :return:
        w: (r, r) float array, whose element is in the range [0, 1]
    """
    u = ((np.arange(0, r) + 1) / r).reshape(-1, 1)  # (r, 1)
    v = ((np.arange(0, r) + 1) / r).reshape(1, -1)  # (1, r)

    if type_idx == 0:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = u @ v
    elif type_idx == 1:
        w = np.exp(-(u ** 0.7 + v ** 0.7))
    elif type_idx == 2:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.25 * (u ** 2 + v ** 2 + u ** 0.5 + v ** 0.5)
    elif type_idx == 3:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 0.5 * (u + v)
    elif type_idx == 4:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-2 * (u ** 2 + v ** 2)))
    elif type_idx == 5:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = 1 / (1 + np.exp(-(np.maximum(u, v) ** 2 + np.minimum(u, v) ** 4)))
    elif type_idx == 6:
        w = np.exp(-np.maximum(u, v) ** 0.75)
    elif type_idx == 7:
        w = np.exp(-0.5 * (np.minimum(u, v) + u ** 0.5 + v ** 0.5))
    elif type_idx == 8:
        u = u[::-1, :]
        v = v[:, ::-1]
        w = np.log(1 + 0.5 * np.maximum(u, v))
    elif type_idx == 9:
        w = np.abs(u - v)
    elif type_idx == 10:
        w = 1 - np.abs(u - v)
    elif type_idx == 101:
        theta = np.pi * 3 / 4
        w = 0.9*np.exp((-(v)**2-(u-1)**2)/alpha**2)+0.9*np.exp((-(u)**2-(v-1)**2)/alpha**2)+0.9*np.exp(-((np.sin(theta)*u+np.cos(theta)*v)/alpha)**2)
    elif type_idx == 11:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), 0.8 * np.ones((r2, r2)))
    elif type_idx == 12:
        r2 = int(r / 2)
        w = np.kron(np.eye(2, dtype=int), np.ones((r2, r2)))
        w = 0.8 * (1 - w)
    
    # parametrized case of graphon 2 and SBM
    elif type_idx == 19:
        w = np.exp(-(u ** 0.7 + v ** 0.7)/alpha)
    elif type_idx == 14:
        w = np.zeros((r, r))
        R = int(r * alpha)
        w[:R, :R] = 0.8
        w[R:, R:] = 0.8
        w[:R, R:] = 0.1
        w[R:, :R] = 0.1
    else:
        raise ValueError('Unknown graphon type')

    np.fill_diagonal(w, 0.)
    return w

if __name__ == '__main__':
    list_of_tensors = []
    for idx in [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        101,
        11,
        12,
        19,
        14
    ]:
        true_graphon = synthesize_graphon(r=32, type_idx=idx, alpha=10)
        for i in range(5):
            saveTensorInImage(true_graphon, f"dataset/sample_{idx}_{i}.png")
    dataset = load_dataset("imagefolder", data_dir="dataset")
    dataset.push_to_hub("mikymatt/graphons", private=True)
    print(dataset)