# src/metrics.py

import numpy as np

def rmse_posicion(estimate, ground_truth):
    """RMSE de posición (x, y)"""
    err = estimate[:, :2] - ground_truth[:, :2]
    return np.sqrt(np.mean(np.sum(err**2, axis=1)))

def ate(estimate, ground_truth):
    """Absolute Trajectory Error"""
    est = estimate[:, :2]
    gt = ground_truth[:, :2]
    est_centered = est - est.mean(axis=0)
    gt_centered = gt - gt.mean(axis=0)
    return np.sqrt(np.mean(np.sum((est_centered - gt_centered)**2, axis=1)))

def rpe(estimate, ground_truth, delta=10):
    """Relative Pose Error"""
    n = len(estimate)
    errores = []
    for i in range(0, n - delta, delta):
        d_est = estimate[i+delta, :2] - estimate[i, :2]
        d_gt = ground_truth[i+delta, :2] - ground_truth[i, :2]
        errores.append(np.linalg.norm(d_est - d_gt))
    return np.mean(errores)

def summary_metrics(estimate, ground_truth, name=""):
    return {
        'name': name,
        'rmse': rmse_posicion(estimate, ground_truth),
        'ate': ate(estimate, ground_truth),
        'rpe': rpe(estimate, ground_truth),
    }