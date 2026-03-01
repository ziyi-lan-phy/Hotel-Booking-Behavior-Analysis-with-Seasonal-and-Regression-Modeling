import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import linregress
from scipy.optimize import curve_fit

#========================== Model Fitting ==========================
def linear_func(x, m, c):
    return m * x + c

def log_func(x, a, b):
    return a * np.log(x) + b

def power_func(x, a, b):
    return a * np.power(x, b)

def quadratic_func(x, c2, c1, c0):
    return c2 * x**2 + c1 * x + c0

def fit_models(x_train, y_train):
    fitted_models = {}

    # Linear
    res = linregress(x_train, y_train)
    fitted_models['Linear'] = (linear_func, (res.slope, res.intercept))

    # Log
    res = linregress(np.log(x_train), y_train)
    fitted_models['Log'] = (log_func, (res.slope, res.intercept))

    # Power
    popt, _ = curve_fit(power_func, x_train, y_train, p0=[0.1, 0.1])
    fitted_models['Power'] = (power_func, popt)

    # Quadratic
    coefs = np.polyfit(x_train, y_train, 2)
    fitted_models['Quadratic'] = (quadratic_func, coefs)

    return fitted_models

def plot_leadtime_regression(x_train, y_train,
                             fitted_models=None,
                             colors=None,
                             days_range=range(30, 148, 10),
                             cutoff_day=None,
                             save_path=None):
    """
    Plot lead time vs cancellation rate regression (training data only).

    Parameters
    ----------
    x_train : array-like
        Lead time values for training data.
    y_train : array-like
        Cancellation rate values for training data.
    fitted_models : dict
        {'Linear': (func, params), ...} fitted functions and parameters.
    colors : dict
        {'Linear': ('blue', '-'), ...} colors and line styles for models.
    days_range : iterable
        List or range of max lead time to compute R² stability.
    cutoff_day : int or None
        Optional vertical line to indicate cutoff day.
    save_path : str or None
        Path to save figure. If None, figure is shown.
    """
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=90)

    # Scatter plot of training data
    ax1.scatter(x_train, y_train, color='gray', alpha=0.3, s=20, label='Training Data')

    # Plot fitted models
    for name, (func, params) in fitted_models.items():
        y_pred_train = func(x_train, *params)
        ax1.plot(x_train, y_pred_train, color=colors[name][0], linestyle=colors[name][1],
                 linewidth=2.5, label=f'{name} Fit')
        r2 = r2_score(y_train, y_pred_train)
        mse = np.mean((y_train - y_pred_train)**2)
        ax1.plot([], [], ' ', label=f'{name} R²: {r2:.3f}, MSE: {mse:.4f}')

    # Optional cutoff line
    if cutoff_day is not None:
        ax1.axvline(cutoff_day, color='black', linestyle=':', label=f'Cutoff at {cutoff_day}')

    ax1.set_xlabel("Lead Time (Days)")
    ax1.set_ylabel("Cancellation Rate")
    ax1.set_title("Training Fit", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize='small', frameon=True, shadow=True)

    # R² stability analysis
    results = {name: [] for name in fitted_models.keys()}
    for d in days_range:
        mask = x_train <= d
        sx, sy = x_train[mask], y_train[mask]
        for name, (func, params) in fitted_models.items():
            try:
                if name == 'Linear':
                    r2 = linregress(sx, sy).rvalue**2
                elif name == 'Log':
                    r2 = linregress(np.log(sx), sy).rvalue**2
                elif name == 'Power':
                    popt, _ = curve_fit(power_func, sx, sy, p0=[0.1,0.1])
                    r2 = r2_score(sy, power_func(sx, *popt))
                elif name == 'Quadratic':
                    coefs = np.polyfit(sx, sy, 2)
                    r2 = r2_score(sy, np.poly1d(coefs)(sx))
                results[name].append(r2)
            except:
                results[name].append(np.nan)

    # Plot R² stability
    for name, scores in results.items():
        ax2.plot(days_range, scores, marker='o', label=name,
                 color=colors[name][0], linestyle=colors[name][1], linewidth=2)

    ax2.set_title("R² Stability over Data Horizon", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Max Lead Time Included (Days)")
    ax2.set_ylabel("R² Score")
    ax2.axhline(0.6, color='black', linestyle=':', alpha=0.5, label='Reliability Threshold')
    ax2.legend(fontsize='small', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_validation_from_fitted(fitted_models, df_val, lead_col='lead_time', target_col='is_canceled',
                                colors=None, cutoff_day=199, show_all_val=True, save_path=None):
    """
    Use pre-fitted models to validate on aggregated validation data and show R²/MSE in legend.

    Parameters
    ----------
    fitted_models : dict
        {'Linear': (func, params), ...}
    df_val : pd.DataFrame
    lead_col : str
    target_col : str
    colors : dict
        {'Linear': ('forestgreen', '--'), ...}
    cutoff_day : int
    show_all_val : bool
    save_path : str or None
        
    Returns
    -------
    results : dict
        {model_name: {'r2_trust':..., 'mse_trust':..., 'r2_all':..., 'mse_all':...}}
    """

    if colors is None:
        colors = {
            'Linear': ('forestgreen', '--'),
            'Log': ('crimson', '-'),
            'Power': ('orange', ':'),
            'Quadratic': ('royalblue', '-.')
        }

    val_trust = df_val[df_val[lead_col] <= cutoff_day].copy()
    val_all   = df_val.copy()
    
    val_trust_stats = val_trust.groupby(lead_col).agg({target_col:'mean'}).reset_index()
    val_all_stats   = val_all.groupby(lead_col).agg({target_col:'mean'}).reset_index()
    
    x_trust = val_trust_stats[lead_col].values
    y_trust = val_trust_stats[target_col].values
    x_all   = val_all_stats[lead_col].values
    y_all   = val_all_stats[target_col].values
    
    fig, ax = plt.subplots(figsize=(10,5), dpi=90)
    
    if show_all_val:
        ax.scatter(x_all, y_all, color='pink', alpha=0.4, s=20, label='Validation (all, avg)')
    ax.scatter(x_trust, y_trust, color='lightcoral', alpha=0.7, s=30, label='Validation (trusted, avg)')
    
    results = {}
    
    for name, (func, params) in fitted_models.items():

        y_pred_trust = func(x_trust, *params)
        y_pred_all   = func(x_all, *params)
        r2_trust = r2_score(y_trust, y_pred_trust)
        mse_trust = mean_squared_error(y_trust, y_pred_trust)
        r2_all   = r2_score(y_all, y_pred_all)
        mse_all  = mean_squared_error(y_all, y_pred_all)
        
        results[name] = {
            'r2_trust': r2_trust,
            'mse_trust': mse_trust,
            'r2_all': r2_all,
            'mse_all': mse_all
        }
        
        ax.plot(x_trust, y_pred_trust, color=colors[name][0], linestyle=colors[name][1],
                linewidth=2, label=f'{name} Fit')
        
        ax.plot([], [], ' ', label=f'{name} R²: {r2_trust:.3f}, MSE: {mse_trust:.4f}')
    
    # cutoff day
    if cutoff_day is not None:
        ax.axvline(cutoff_day, color='black', linestyle=':', label=f'Cutoff at {cutoff_day}')
    
    ax.set_xlabel("Lead Time (Days)")
    ax.set_ylabel("Cancellation Rate")
    ax.set_title("Validation Check: Predicted vs Actual", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize='small', frameon=True, shadow=True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return results

