import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import linregress
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from fun import compute_survival_function

# ========= guests ==========
def plot_cancellation_guests(df_plot, save_path=None):
    """
    Plot cancellation rate by Total Guests, Adults, Children, Babies in 1 row, 4 columns.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Booking dataset with columns:
        ['total_guests', 'adults', 'children', 'babies', 'is_canceled']
    """
    # --- Prepare values and rates ---
    guest_values = sorted(df_plot['total_guests'].unique())
    guest_counts = df_plot['total_guests'].value_counts().reindex(guest_values, fill_value=0)
    guest_cancel_counts = df_plot[df_plot['is_canceled'] == 1]['total_guests'].value_counts().reindex(guest_values, fill_value=0)
    guest_cancel_rate = guest_cancel_counts / guest_counts

    adults_values = sorted(df_plot['adults'].unique())
    adults_cancel_rate = df_plot.groupby('adults')['is_canceled'].mean().reindex(adults_values, fill_value=0)

    children_values = sorted(df_plot['children'].unique())
    children_cancel_rate = df_plot.groupby('children')['is_canceled'].mean().reindex(children_values, fill_value=0)

    babies_values = sorted(df_plot['babies'].unique())
    babies_cancel_rate = df_plot.groupby('babies')['is_canceled'].mean().reindex(babies_values, fill_value=0)

    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20,5))  # 1行4列

    sns.barplot(x=guest_values, y=guest_cancel_rate.values, color='#FF6F61', edgecolor='black', alpha=0.7, ax=axes[0])
    axes[0].set_title("Cancellation Rate by Total Guests", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Total Guests")
    axes[0].set_ylabel("Cancellation Rate")
    axes[0].set_ylim(0,1)
    axes[0].grid(True, linestyle='--', alpha=0.5)

    sns.barplot(x=adults_values, y=adults_cancel_rate.values, color='#1F77B4', edgecolor='black', alpha=0.7, ax=axes[1])
    axes[1].set_title("Cancellation Rate by Adults", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Adults")
    axes[1].set_ylabel("Cancellation Rate")
    axes[1].set_ylim(0,1)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    sns.barplot(x=children_values, y=children_cancel_rate.values, color='#2CA02C', edgecolor='black', alpha=0.7, ax=axes[2])
    axes[2].set_title("Cancellation Rate by Children", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Children")
    axes[2].set_ylabel("Cancellation Rate")
    axes[2].set_ylim(0,1)
    axes[2].grid(True, linestyle='--', alpha=0.5)

    sns.barplot(x=babies_values, y=babies_cancel_rate.values, color='#9467BD', edgecolor='black', alpha=0.7, ax=axes[3])
    axes[3].set_title("Cancellation Rate by Babies", fontsize=12, fontweight='bold')
    axes[3].set_xlabel("Babies")
    axes[3].set_ylabel("Cancellation Rate")
    axes[3].set_ylim(0,1)
    axes[3].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # --- Save figure if path is provided ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.show()

# ========== nights ==========
def plot_total_nights_distribution(df, column='total_nights', max_nights=None, save_path=None):
    """
    Plot distribution of total nights:
    - Linear histogram
    - Log-scale histogram
    - Survival function (≥ N nights)
    
    Parameters:
        df: pd.DataFrame or pd.Series
        column: str, column name in df if df is DataFrame
        max_nights: int, optional upper limit for filtering nights (like 95% quantile)
    """
    
    if isinstance(df, pd.DataFrame):
        df_plot = df[column]
    else:
        df_plot = df.copy()
    
    if max_nights is not None:
        df_plot = df_plot[df_plot <= max_nights]
    
    bins = range(1, df_plot.max()+2)
    
    fig, axes = plt.subplots(1, 3, figsize=(18,5))
    
    # --- Linear histogram ---
    sns.histplot(df_plot, bins=bins, kde=False, stat='probability', 
                 color='#FF6F61', edgecolor='black', alpha=0.7, ax=axes[0])
    axes[0].axvline(df_plot.mean(), color='#1F77B4', linestyle='--', linewidth=2,
                     label=f'Mean: {df_plot.mean():.2f}')
    axes[0].axvline(df_plot.median(), color='#2CA02C', linestyle='-', linewidth=2,
                     label=f'Median: {df_plot.median():.0f}')
    axes[0].set_title("Linear Scale", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Total Nights")
    axes[0].set_ylabel("Probability")
    axes[0].set_ylim(0,1)
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- Log-scale histogram ---
    sns.histplot(df_plot, bins=bins, kde=False, stat='probability', 
                 color='#FF6F61', edgecolor='black', alpha=0.7, ax=axes[1])
    axes[1].axvline(df_plot.mean(), color='#1F77B4', linestyle='--', linewidth=2)
    axes[1].axvline(df_plot.median(), color='#2CA02C', linestyle='-', linewidth=2)
    axes[1].set_yscale('log')
    axes[1].set_title("Log Scale", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Total Nights")
    axes[1].set_ylabel("Probability (log)")
    axes[1].grid(True, which='both', linestyle='--', alpha=0.5)

    # --- Survival Function ---
    pmf, sf = compute_survival_function(df_plot)
    
    sns.histplot(df_plot, bins=bins, kde=False, stat='probability', 
                 color='#FF6F61', edgecolor='black', alpha=0.7, ax=axes[2])
    axes[2].plot(sf.index, sf.values, color='steelblue', marker='o', linewidth=2, label='≥ N Nights')
    axes[2].set_title("Survival Function", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Total Nights")
    axes[2].set_ylabel("Probability / ≥ N")
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    # --- Save figure if path is provided ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.show()

    plt.show()

def plot_cancellation_nights_week_end(df, max_total_nights=30, save_path=None):
    """
    Plot cancellation rates by weeknights, weekend nights, and total nights.
    
    Parameters
    ----------
    df : pd.DataFrame
        Booking dataset with columns ['adults', 'children', 'stays_in_week_nights',
                                      'stays_in_weekend_nights', 'total_nights', 'is_canceled']
    max_total_nights : int
        Maximum number of total nights to include (for filtering long stays)
    save_path : str, optional
        If provided, saves the figure to this path.
    """
    
    # --- Filter dataset ---
    df_filtered = df[
        (df['adults'] <= 4) & 
        (df['children'] <= 3) & 
        (df['total_nights'] <= max_total_nights)
    ].copy()
    
    # --- Compute cancellation rates ---
    df_week = df_filtered[df_filtered['stays_in_week_nights'] > 0]
    df_weekend = df_filtered[df_filtered['stays_in_weekend_nights'] > 0]

    week_nights_cancellation = df_week.groupby('stays_in_week_nights')['is_canceled'].mean()
    weekend_nights_cancellation = df_weekend.groupby('stays_in_weekend_nights')['is_canceled'].mean()

    stay_counts = df_filtered.groupby('total_nights').size()
    cancel_counts = df_filtered[df_filtered['is_canceled'] == 1].groupby('total_nights').size()
    cancel_rate_total = (cancel_counts / stay_counts).fillna(0)
    
    # --- Plotting ---
    fig = plt.figure(figsize=(14,10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0,0])
    week_nights_cancellation.plot(kind='bar', color='#FFB347', edgecolor='black', alpha=0.7, ax=ax0)
    ax0.set_title("Cancellation Rate by Weeknights", fontsize=14, fontweight='bold')
    ax0.set_xlabel("Weeknights")
    ax0.set_ylabel("Cancellation Rate")
    ax0.set_ylim(0,1)
    ax0.grid(axis='y', linestyle='--', alpha=0.5)

    ax1 = fig.add_subplot(gs[0,1])
    weekend_nights_cancellation.plot(kind='bar', color='#6A5ACD', edgecolor='black', alpha=0.7, ax=ax1)
    ax1.set_title("Cancellation Rate by Weekend Nights", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Weekend Nights")
    ax1.set_ylabel("Cancellation Rate")
    ax1.set_ylim(0,1)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    ax2 = fig.add_subplot(gs[1,:])
    cancel_rate_total.plot(kind='bar', color='#FF6F61', edgecolor='black', alpha=0.7, ax=ax2)
    ax2.set_title(f"Cancellation Rate by Total Nights (≤{max_total_nights} nights)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Total Nights")
    ax2.set_ylabel("Cancellation Rate")
    ax2.set_xticks(range(len(cancel_rate_total)))
    ax2.set_xticklabels(cancel_rate_total.index, rotation=0)
    ax2.set_ylim(0,1)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()

    # --- Save figure if path is provided ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)

def plot_sf_total_nights(data_plot, ax=None):
    """
    Plot Survival Function of total nights with exponential fit,
    and show fit formula, parameters, and R² on the plot.
    If ax is provided, draw on that axis; else create new figure.
    """
    import matplotlib.pyplot as plt
    import os
    from scipy.optimize import curve_fit
    import numpy as np

    data_plot = data_plot.copy()
    pmf, sf = compute_survival_function(data_plot)
    x = sf.index.values
    y = sf.values

    def exp_func(x, a, b):
        return a * np.exp(-b * x)

    popt, _ = curve_fit(exp_func, x, y, p0=[1, 0.1])
    y_exp = exp_func(x, *popt)

    residuals = y - y_exp
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res/ss_tot

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))

    ax.scatter(x, y, color='steelblue', label='SF data (≥ N nights)')
    ax.plot(x, y_exp, color='red', linestyle='-', label='Exponential fit')
    ax.set_xlabel("Total Nights")
    ax.set_ylabel("Survival Function ≥ N")
    ax.set_title("Survival Function Fit", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    textstr = f"SF(N) = a·exp(-b·N)\na = {popt[0]:.3f}\nb = {popt[1]:.3f}\nR² = {r2:.3f}"
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    ax.legend()
    return ax

def plot_cancellation_nights(df, max_nights=14, ax=None, max_marker_size=200):
    """
    Scatter plot: cancellation rate vs total nights
    - Point size = number of bookings
    - Optional linear fit line
    - Can plot on provided ax
    """
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    df_plot = df[df['total_nights'] <= max_nights]

    nights_values = sorted(df_plot['total_nights'].unique())
    nights_counts = df_plot['total_nights'].value_counts().reindex(nights_values, fill_value=0)
    nights_cancel_counts = df_plot[df_plot['is_canceled']==1]['total_nights'].value_counts().reindex(nights_values, fill_value=0)
    nights_cancel_rate = nights_cancel_counts / nights_counts

    sizes = (nights_counts.values / nights_counts.values.max()) * max_marker_size
    sizes = np.sqrt(sizes) * 20

    if ax is None:
        fig, ax = plt.subplots(figsize=(8,5))

    scatter = ax.scatter(nights_values, nights_cancel_rate.values, 
                         s=sizes, color='steelblue', alpha=0.6, edgecolor='black', zorder=2)

    # X = np.array(nights_values).reshape(-1,1)
    # y = nights_cancel_rate.values
    # lr = LinearRegression()
    # lr.fit(X, y)
    # y_fit = lr.predict(X)
    # r2 = lr.score(X, y)

    # ax.plot(nights_values, y_fit, color='green', linestyle='--', linewidth=2, label=f'Linear fit R²={r2:.3f}', zorder=3)

    for marker_count in [int(nights_counts.max()*0.25), int(nights_counts.max()*0.5), int(nights_counts.max()*0.9)]:
        ax.scatter([], [], s=np.sqrt(marker_count / nights_counts.max()) * 20, 
                   color='steelblue', alpha=0.6, edgecolor='black', label=f'{marker_count} bookings')

    ax.set_xlabel("Total Nights")
    ax.set_ylabel("Cancellation Rate")
    ax.set_ylim(0,1)
    ax.set_title("Cancellation Rate vs Total Nights", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(frameon=True, fontsize='small')

    return ax

# ========= lead_time ==========
def plot_leadtime_decay(df_mask, pct_threshold=0.9, zoom_range=(150, 250), save_path=None):
    """
    Plot booking volume decay and long-tail detail by lead_time.
    
    Parameters
    ----------
    df_mask : pd.DataFrame
        DataFrame with columns ['lead_time', 'booking_count'].
    pct_threshold : float, default 0.9
        Cumulative percentage threshold to define cutoff_day.
    zoom_range : tuple(int, int), default (150, 250)
        x-range for long-tail zoom plot.
    save_path : str or None
        Path to save figure, if None shows figure.
    """
    # --- Prepare cumulative percentage & cutoff day ---
    df_plot = df_mask.copy()
    df_plot = df_plot.sort_values('lead_time').reset_index(drop=True)
    df_plot['cumulative_booking'] = df_plot['booking_count'].cumsum()
    total_booking = df_plot['booking_count'].sum()
    df_plot['pct'] = df_plot['cumulative_booking'] / total_booking

    # cutoff day where cumulative ≥ pct_threshold
    cutoff_day = df_plot[df_plot['pct'] >= pct_threshold]['lead_time'].iloc[0]

    # data for cutoff zone
    df_cutoff_zone = df_plot[df_plot['lead_time'] <= cutoff_day]

    # min booking count in cutoff zone
    n_min = df_cutoff_zone['booking_count'].min()
    lead_min_n = df_cutoff_zone.loc[df_cutoff_zone['booking_count'] == n_min, 'lead_time'].iloc[0]

    # booking count at cutoff_day
    n_pct = df_plot[df_plot['lead_time'] == cutoff_day]['booking_count'].iloc[0]

    # zoom range for long-tail plot
    df_zoom = df_plot[(df_plot['lead_time'] >= zoom_range[0]) & (df_plot['lead_time'] <= zoom_range[1])]

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=90)
    color_main = 'teal'
    color_cutoff = 'orange'
    color_n = 'red'

    # Linear plot
    sns.lineplot(data=df_plot, x='lead_time', y='booking_count', ax=ax1, color=color_main, linewidth=2)
    ax1.fill_between(df_plot['lead_time'], df_plot['booking_count'], color=color_main, alpha=0.1)
    ax1.axvline(x=cutoff_day, color=color_cutoff, linestyle='--', 
                label=f'{int(pct_threshold*100)}% Cumulative at Day {int(cutoff_day)}, n={int(n_pct)}')
    ax1.set_title("Booking Volume Decay (Linear)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Lead Time (Days)")
    ax1.set_ylabel("Number of Bookings")
    ax1.grid(True, alpha=0.2)
    ax1.legend(fontsize='small', frameon=True)
    sns.despine(ax=ax1)

    # Log-scale long-tail
    ax2.set_yscale('log')
    ax2.scatter(lead_min_n, n_min, color=color_n, s=100, zorder=5,
                label=f'n_min={int(n_min)} at day {int(lead_min_n)}')
    ax2.scatter(cutoff_day, n_pct, color=color_cutoff, s=100, zorder=5,
                label=f'n_{int(pct_threshold*100)}%={int(n_pct)} at day {int(cutoff_day)}')
    sns.lineplot(data=df_zoom, x='lead_time', y='booking_count', ax=ax2, color=color_main, linewidth=2)
    ax2.fill_between(df_zoom['lead_time'], df_zoom['booking_count'], color=color_main, alpha=0.1)
    ax2.axvline(x=cutoff_day, color=color_cutoff, linestyle='--',
                label=f'{int(pct_threshold*100)}% Cumulative at Day {int(cutoff_day)}')
    ax2.axhline(y=n_min, color=color_n, linestyle='--', label=f'n_min={int(n_min)}')
    ax2.set_title("Long-tail Detail (Log Scale)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Lead Time (Days)")
    ax2.set_ylabel("Bookings (Log Scale)")
    ax2.grid(True, alpha=0.2)
    ax2.legend(fontsize='small', frameon=True)
    sns.despine(ax=ax2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def plot_leadtime_cancel(df, lead_col='lead_time', cancel_col='is_canceled', 
                         min_lead=0, max_lead=500, cutoff_day=350, 
                         figsize=(10,5), size_range=(20,500), palette='viridis',
                         save_path=None):
    """
    Plot a scatter of lead time vs cancellation rate, with point size proportional to number of bookings.
    Includes a cutoff line inside the plot and legend inside the figure. Can save the figure.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    lead_col : str
        Column name for lead time
    cancel_col : str
        Column name for cancellation flag
    min_lead : int
        Minimum lead time to include
    max_lead : int
        Maximum lead time to include
    cutoff_day : int
        Lead time cutoff for data reliability
    figsize : tuple
        Figure size
    size_range : tuple
        Range of scatter point sizes
    palette : str
        Color palette
    save_path : str or None
        File path to save the figure. If None, the figure is not saved.
    """

    # --- Aggregate data ---
    lead_time_counts = df.groupby(lead_col).agg(
        booking_count=(cancel_col, 'count'),
        cancel_rate=(cancel_col, 'mean')
    ).reset_index()

    lead_time_counts = lead_time_counts[
        (lead_time_counts[lead_col] > min_lead) & 
        (lead_time_counts[lead_col] < max_lead)
    ]

    # --- Plot ---
    plt.figure(figsize=figsize)
    ax = sns.scatterplot(
        data=lead_time_counts,
        x=lead_col, y='cancel_rate',
        size='booking_count', hue='booking_count',
        palette=palette, alpha=0.5, sizes=size_range
    )

    # Cutoff line
    plt.axvline(x=cutoff_day, color='orange', linestyle='--', 
                label=f'Data Reliability Cut-off ({cutoff_day} days)')

    # Legend inside the plot
    ax.legend(title="Booking Count", loc='upper left', fontsize='small', 
              title_fontsize='small', frameon=True)

    plt.title("Cancel Rate vs Lead Time: Volume-Weighted Dispersion", fontsize=14, fontweight='bold')
    plt.xlabel("Lead Time (Days)")
    plt.ylabel("Cancellation Rate (0-1)")
    plt.grid(True, alpha=0.2)

    # --- Save figure ---
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_validation_fit(
    x_train, y_train, 
    df_val, lead_col='lead_time', target_col='is_canceled',
    fit_func=None, p0=None,
    cutoff_day=199,
    show_all_val=True,
    save_path=None
):
    """
    Plot training fit and validation aggregated points with metrics.
    
    Parameters
    ----------
    x_train, y_train : np.ndarray
        Training data arrays.
    df_val : pd.DataFrame
        Validation DataFrame with lead_col and target_col.
    lead_col : str
        Column name for lead time.
    target_col : str
        Column name for cancellation target.
    fit_func : callable
        Function to fit, e.g., power_func(x, a, b). If None, defaults to power_func.
    p0 : list
        Initial guess for curve_fit.
    cutoff_day : int
        Cutoff day for trusted validation data.
    show_all_val : bool
        Whether to plot all validation points.
    save_path : str or None
        Path to save figure.
        
    Returns
    -------
    popt, r2_trust, mse_trust, r2_all, mse_all
    """
    
    if fit_func is None:
        def fit_func(x, a, b):
            return a * np.power(x, b)
    if p0 is None:
        p0 = [0.5, 0.5]
    
    # --- Fit training data ---
    popt, _ = curve_fit(fit_func, x_train, y_train, p0=p0)
    
    # --- Validation data aggregation ---
    val_trust = df_val[df_val[lead_col] <= cutoff_day].copy()
    val_all   = df_val.copy()
    
    val_trust_stats = val_trust.groupby(lead_col).agg(is_canceled=('is_canceled','mean')).reset_index()
    x_val_trust = val_trust_stats[lead_col].values
    y_val_trust = val_trust_stats['is_canceled'].values

    val_all_stats = val_all.groupby(lead_col).agg(is_canceled=('is_canceled','mean')).reset_index()
    x_val_all = val_all_stats[lead_col].values
    y_val_all = val_all_stats['is_canceled'].values

    # --- Metrics ---
    y_val_trust_pred = fit_func(x_val_trust, *popt)
    r2_val_trust = r2_score(y_val_trust, y_val_trust_pred)
    mse_val_trust = mean_squared_error(y_val_trust, y_val_trust_pred)

    y_val_all_pred = fit_func(x_val_all, *popt)
    r2_val_all = r2_score(y_val_all, y_val_all_pred)
    mse_val_all = mean_squared_error(y_val_all, y_val_all_pred)
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10,5), dpi=90)
    
    if show_all_val:
        ax.scatter(x_val_all, y_val_all, color='pink', alpha=0.3, s=20, label='Validation (all, avg)')
    
    ax.scatter(x_val_trust, y_val_trust, color='lightcoral', alpha=0.7, s=30, label='Validation (trusted, avg)')
    
    x_sorted = np.sort(x_train)
    ax.plot(x_sorted, fit_func(x_sorted, *popt), color='orange', linewidth=2, label='Fit (train)')
    
    ax.plot(np.sort(x_val_trust), fit_func(np.sort(x_val_trust), *popt), color='blue', linestyle='--', linewidth=2, label='Prediction (trusted)')
    
    # cutoff_day
    ax.axvline(x=cutoff_day, color='black', linestyle=':', label=f'Cutoff at {cutoff_day} days')
    
    param_str = ', '.join([f'{v:.3f}' for v in popt])
    textstr = '\n'.join((
        f'Fit: {fit_func.__name__}({param_str})',
        f'R² (trusted) = {r2_val_trust:.3f}',
        f'MSE (trusted) = {mse_val_trust:.4f}',
        f'R² (all) = {r2_val_all:.3f}',
        f'MSE (all) = {mse_val_all:.4f}'
    ))
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel("Lead Time (Days)")
    ax.set_ylabel("Cancellation Rate")
    ax.set_title("Validation Check: Aggregated Cancellation Rate vs Fit")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize='small', frameon=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return popt, r2_val_trust, mse_val_trust, r2_val_all, mse_val_all

