import numpy as np
from tqdm import tqdm
import pandas as pd


n_examples = []
means = []
stds = []

for i in tqdm(range(1, 120)):

    temp_df = pd.read_hdf(r'B:\MedleyDB\dataset.h5', key=str(i))

    x_mag = np.array(temp_df['raw magnitude'].tolist())
    y_mag = np.array(temp_df['stem magnitude'].tolist())
    # x_phase = np.array(temp_df['raw phase'].tolist())
    # y_phase = np.array(temp_df['stem phase'].tolist())

    # x_mag = np.log10(x_mag + 1e-10)
    # y_mag = np.log10(y_mag + 1e-10)

    n_examples.append(np.size(x_mag))
    n_examples.append(np.size(y_mag))
    means.append(np.mean(x_mag))
    means.append(np.mean(y_mag))
    stds.append(np.std(x_mag))
    stds.append(np.std(y_mag))

n_examples = np.asarray(n_examples)
means = np.asarray(means)
stds = np.asarray(stds)

sumX = np.multiply(n_examples, means)
sumX2 = np.add(np.multiply(np.square(stds), n_examples - 1), np.divide(np.square(sumX), n_examples))
tn = np.sum(n_examples, dtype=np.uint64)
tx = np.sum(sumX)
txx = np.sum(sumX2)
mean = tx/tn
std = np.sqrt((txx - tx**2 / tn) / (tn - 1))

print("mean = ", mean, "\n std = ", std)
