# src/utils.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_path(name):
    df = pd.read_csv(f"data/{name}_gt.csv")
    with open(f"data/{name}_meta.json") as f:
        meta = json.load(f)
    return df, meta

def plot_path(gt, estimates, title="Trayectorias"):
    plt.figure(figsize=(8, 8))
    plt.plot(gt['x'], gt['y'], 'k-', lw=2, label='Ground truth')
    for nombre, est in estimates.items():
        plt.plot(est[:, 0], est[:, 1], '--', label=nombre, alpha=0.8)
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.show()

def plot_error(gt, estimates):
    """Error de posición a lo largo del tiempo."""
    plt.figure(figsize=(10, 4))
    for nombre, est in estimates.items():
        err = np.linalg.norm(est[:, :2] - gt[['x', 'y']].values, axis=1)
        plt.plot(gt['t'], err, label=nombre)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Error de posición (m)')
    plt.legend()
    plt.grid(True)
    plt.title('Error de posición a lo largo del tiempo')
    plt.show()