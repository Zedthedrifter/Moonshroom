import pandas as pd
import numpy as np
from scipy.stats import circmean, circstd, vonmises
from astropy.stats import rayleightest, vonmisesmle

def LGP_circular_test(all_df, species_list, years=None):
    """
    Test whether lunar gravitational peaks (LGP) are associated with a
    non-random phase shift in peak harvest day, using circular statistics.
    Approach:
      1. Build a week-level dataframe with the peak harvest day (as an angle)
         and the LGP day (as an angle) for each cycle.
      2. For each species, compute the angular distance between peak day and
         LGP day, and test whether the distribution of distances is uniform
         (Rayleigh test) or concentrated around a mean angle (von Mises MLE).
      3. Optionally, fit a circular-circular regression (peak_day ~ lgp_day)
         using the linearised sine/cosine model described in the literature.
    Parametes
    ----------
    all_df : dict
        {year: {'df': DataFrame}} with columns 'date', 'dayofweek',
        'has_gravity_peak', and species columns.
    species_list : list of str
    years : iterable of int, optional
    Returns
    -------
    results_df : DataFrame
        Per-species circular statistics and Rayleigh test p-values.
    """
    if years is None:
        years = sorted(all_df.keys())
    # ---- 1. Build week-level dataframe with angles ----
    rows = []
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        # Wednesday = 0 … Tuesday = 6
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        df['cycle_start'] = df['date'] - pd.to_timedelta(df['cycle_offset'], unit='D')
        # LGP day within the cycle (first day with gravity peak, or NaN)
        lgp_day_by_cycle = {}
        for cycle, group in df.groupby('cycle_start'):
            peaks = group.loc[group['has_gravity_peak'] == True, 'cycle_offset']
            lgp_day_by_cycle[cycle] = int(peaks.min()) if not peaks.empty else np.nan
        for sp in species_list:
            if sp not in df.columns:
                continue
            sub = df[['cycle_start', 'cycle_offset', sp]].dropna(subset=[sp])
            if sub.empty:
                continue
            for cycle, group in sub.groupby('cycle_start'):
                non_zero = group[group[sp] > 0]
                if len(non_zero) < 3:
                    continue  # same filter as before
                m = non_zero[sp].max()
                tied = non_zero.loc[non_zero[sp] == m, 'cycle_offset'].astype(int).tolist()
                peak_day = min(tied)  # 0=Wed … 6=Tue
                lgp_day = lgp_day_by_cycle.get(cycle, np.nan)
                # Convert day indices (0–6) to angles in radians (period = 7)
                peak_angle = 2 * np.pi * peak_day / 7
                lgp_angle = 2 * np.pi * lgp_day / 7 if not np.isnan(lgp_day) else np.nan
                # Angular distance (shortest path on the circle)
                if not np.isnan(lgp_angle):
                    diff = np.abs(peak_angle - lgp_angle)
                    ang_dist = np.minimum(diff, 2*np.pi - diff)
                else:
                    ang_dist = np.nan
                rows.append({
                    'species': sp, 'year': yr, 'cycle_start': cycle,
                    'peak_day': peak_day, 'lgp_day': lgp_day,
                    'peak_angle': peak_angle, 'lgp_angle': lgp_angle,
                    'ang_dist': ang_dist,
                })
    week_df = pd.DataFrame(rows)
    if week_df.empty:
        raise ValueError("No weeks passed the ≥3 non-zero filter.")
    # ---- 2. Circular tests per species ----
    out = []
    for sp in species_list:
        sub = week_df[week_df['species'] == sp].copy()
        if sub.empty:
            continue
        # Rayleigh test on peak angles (is the peak day non-uniform?)
        peak_angles = sub['peak_angle'].dropna().values
        if len(peak_angles) >= 2:
            p_rayleigh = rayleightest(peak_angles)
            mean_peak_angle, kappa_peak = vonmisesmle(peak_angles)
            # Convert mean angle back to day index (0–6)
            mean_peak_day = (mean_peak_angle * 7 / (2*np.pi)) % 7
        else:
            p_rayleigh = np.nan
            mean_peak_day = np.nan
            kappa_peak = np.nan
        # Angular distance between peak and LGP (only LGP weeks)
        lgp_weeks = sub.dropna(subset=['lgp_angle'])
        if len(lgp_weeks) >= 2:
            dists = lgp_weeks['ang_dist'].dropna().values
            p_rayleigh_dist = rayleightest(dists) if len(dists) >= 2 else np.nan
            mean_dist, kappa_dist = vonmisesmle(dists) if len(dists) >= 2 else (np.nan, np.nan)
            mean_dist_days = mean_dist * 7 / (2*np.pi)
        else:
            p_rayleigh_dist = np.nan
            mean_dist_days = np.nan
            kappa_dist = np.nan
        out.append({
            'species': sp,
            'n_weeks': len(sub),
            'n_lgp_weeks': len(lgp_weeks),
            'mean_peak_day': round(mean_peak_day, 2) if not np.isnan(mean_peak_day) else np.nan,
            'kappa_peak': round(kappa_peak, 3) if not np.isnan(kappa_peak) else np.nan,
            'rayleigh_p_peak': p_rayleigh,
            'mean_ang_dist_days': round(mean_dist_days, 2) if not np.isnan(mean_dist_days) else np.nan,
            'kappa_dist': round(kappa_dist, 3) if not np.isnan(kappa_dist) else np.nan,
            'rayleigh_p_dist': p_rayleigh_dist,
        })
    return pd.DataFrame(out), week_df


# ---- Usage ----
results_circ, week_data = LGP_circular_test(all_df, species_list)
print(results_circ.to_string(index=False, float_format='%.3f'))
