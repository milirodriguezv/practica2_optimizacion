# src/sensor_simulator.py
"""
Simula las medidas de GPS, IMU y encoders a partir del ground truth.
"""
import numpy as np
import pandas as pd

class SensorSimulator:
    def __init__(self, config=None, seed=42):
        self.cfg = config or {
            'gps': {
                'sigma_xy': 1.5,       # Ruido
                'outlier_prob': 0.05,  # Medidas erróneas
                'outlier_scale': 10.0, # Desviación de las medidas erróneas
            },
            'imu': {
                'sigma_a': 0.1,        # Ruido del acelerómetro
                'sigma_omega': 0.02,   # Ruido del giroscopio
                'bias_a_initial': 0.05, # Deriva del acelerómetro
                'bias_omega_initial': 0.01, # Deriva del giroscopio
                'bias_random_walk': 1e-4, # Deriva a largo plazo
            },
            'encoders': {
                'sigma_delta': 0.02,   # Ruido
                'scale': 1.02,        # Error por calibración
            }
        }
        self.rng = np.random.default_rng(seed)
    
    def sim_gps(self, df_gt):
        n = len(df_gt)
        z = np.zeros((n, 2))
        is_outlier = self.rng.random(n) < self.cfg['gps']['outlier_prob']
        ruido = self.rng.normal(0, self.cfg['gps']['sigma_xy'], (n, 2))
        ruido[is_outlier] *= self.cfg['gps']['outlier_scale']
        z[:, 0] = df_gt['x'].values + ruido[:, 0]
        z[:, 1] = df_gt['y'].values + ruido[:, 1]
        return pd.DataFrame({
            't': df_gt['t'].values,
            'x_gps': z[:, 0],
            'y_gps': z[:, 1],
            'is_outlier': is_outlier.astype(int)
        })
    
    def sim_imu(self, df_gt):
        n = len(df_gt)
        dt = df_gt['t'].iloc[1] - df_gt['t'].iloc[0]
        
        bias_a = np.zeros(n)
        bias_omega = np.zeros(n)
        bias_a[0] = self.cfg['imu']['bias_a_initial']
        bias_omega[0] = self.cfg['imu']['bias_omega_initial']
        for i in range(1, n):
            bias_a[i] = bias_a[i-1] + self.rng.normal(0, self.cfg['imu']['bias_random_walk'])
            bias_omega[i] = bias_omega[i-1] + self.rng.normal(0, self.cfg['imu']['bias_random_walk'])
        
        ax_real = np.gradient(df_gt['vx'].values, dt)
        ay_real = np.gradient(df_gt['vy'].values, dt)
        omega_real = df_gt['omega'].values
        
        ax_med = ax_real + bias_a + self.rng.normal(0, self.cfg['imu']['sigma_a'], n)
        ay_med = ay_real + bias_a + self.rng.normal(0, self.cfg['imu']['sigma_a'], n)
        omega_med = omega_real + bias_omega + self.rng.normal(0, self.cfg['imu']['sigma_omega'], n)
        
        return pd.DataFrame({
            't': df_gt['t'].values,
            'ax_imu': ax_med,
            'ay_imu': ay_med,
            'omega_imu': omega_med,
            'bias_a': bias_a,
            'bias_omega': bias_omega
        })
    
    def sim_encoders(self, df_gt):
        n = len(df_gt)
        dt = df_gt['t'].iloc[1] - df_gt['t'].iloc[0]
        
        dx = np.diff(df_gt['x'].values, prepend=df_gt['x'].values[0])
        dy = np.diff(df_gt['y'].values, prepend=df_gt['y'].values[0])
        ds_real = np.sqrt(dx**2 + dy**2)
        
        ds_med = ds_real * self.cfg['encoders']['scale'] + \
                 self.rng.normal(0, self.cfg['encoders']['sigma_delta'], n)
        
        dtheta = np.diff(df_gt['theta'].values, prepend=df_gt['theta'].values[0])
        dtheta_med = dtheta + self.rng.normal(0, 0.01, n)
        
        return pd.DataFrame({
            't': df_gt['t'].values,
            'ds_enc': ds_med,
            'dtheta_enc': dtheta_med
        })
    
    def sim_sensors(self, df_gt):
        return {
            'gps': self.sim_gps(df_gt),
            'imu': self.sim_imu(df_gt),
            'encoders': self.sim_encoders(df_gt)
        }