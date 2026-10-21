import pandas as pd
import numpy as np
from astropy.stats import rayleightest, vonmisesmle
import statsmodels.api as sm

def LGP_circular_test(all_df, species_list, years=None):
    """
    Test whether lunar gravitational peaks (LGP) are associated with a
    phase shift in peak harvest day, using circular statistics.

    LGP day is defined as the day between the last negative deldot and the
    first positive deldot within the cycle (moon closest to Earth).

    Per species:
      - Rayleigh test on peak-day angles (is the peak day non-uniform?)
      - Rayleigh test on angular distance between peak day and LGP day
      - Circular-circular regression: peak_angle ~ sin/cos(lgp_angle)
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

        # ---- Identify LGP day per cycle from deldot sign change ----
        lgp_day_by_cycle = {}
        for cycle, group in df.groupby('cycle_start'):
            group = group.sort_values('date')
            deldot = group['deldot (km/s)'].values
            offsets = group['cycle_offset'].values
            lgp_day = np.nan
            for i in range(len(deldot) - 1):
                if deldot[i] < 0 and deldot[i + 1] > 0:
                    # Moon closest to Earth: between day i and day i+1.
                    # Assign the LGP to the midpoint (half-day precision).
                    lgp_day = (offsets[i] + offsets[i + 1]) / 2.0
                    break
            lgp_day_by_cycle[cycle] = lgp_day

        for sp in species_list:
            if sp not in df.columns:
                continue
            sub = df[['cycle_start', 'cycle_offset', sp]].dropna(subset=[sp])
            if sub.empty:
                continue
            for cycle, group in sub.groupby('cycle_start'):
                non_zero = group[group[sp] > 0]
                if len(non_zero) < 3:
                    continue  # same ≥3 non-zero filter as before
                m = non_zero[sp].max()
                tied = non_zero.loc[non_zero[sp] == m, 'cycle_offset'].astype(int).tolist()
                peak_day = min(tied)  # 0=Wed … 6=Tue
                lgp_day = lgp_day_by_cycle.get(cycle, np.nan)

                peak_angle = 2 * np.pi * peak_day / 7
                lgp_angle = 2 * np.pi * lgp_day / 7 if not np.isnan(lgp_day) else np.nan

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

        # (a) Rayleigh test on peak angles
        peak_angles = sub['peak_angle'].dropna().values
        if len(peak_angles) >= 2:
            p_rayleigh_peak = rayleightest(peak_angles)
            mean_peak_angle, kappa_peak = vonmisesmle(peak_angles)
            mean_peak_day = (mean_peak_angle * 7 / (2*np.pi)) % 7
        else:
            p_rayleigh_peak = np.nan
            mean_peak_day = np.nan
            kappa_peak = np.nan

        # (b) Rayleigh test on angular distance (LGP weeks only)
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

        # (c) Circular–circular regression: peak_angle ~ sin/cos(lgp_angle)
        #     Only on LGP weeks (where lgp_angle is defined)
        circ_reg_p = np.nan
        circ_reg_coef_sin = np.nan
        circ_reg_coef_cos = np.nan
        circ_reg_r2 = np.nan
        if len(lgp_weeks) >= 5:
            y = lgp_weeks['peak_angle'].values
            x = lgp_weeks['lgp_angle'].values
            # Linearised circular regression: y ~ a*sin(x) + b*cos(x) + c
            X = np.column_stack([np.sin(x), np.cos(x), np.ones(len(x))])
            try:
                model = sm.OLS(y, X).fit()
                circ_reg_coef_sin = model.params[0]
                circ_reg_coef_cos = model.params[1]
                circ_reg_r2 = model.rsquared
                # F-test for the joint significance of sin and cos terms
                # (H0: a = b = 0, i.e., no circular relationship)
                f_test = model.f_test("x1 = 0, x2 = 0")
                circ_reg_p = float(f_test.pvalue)
            except Exception:
                pass

        out.append({
            'species': sp,
            'n_weeks': len(sub),
            'n_lgp_weeks': len(lgp_weeks),
            'mean_peak_day': round(mean_peak_day, 2) if not np.isnan(mean_peak_day) else np.nan,
            'kappa_peak': round(kappa_peak, 3) if not np.isnan(kappa_peak) else np.nan,
            'rayleigh_p_peak': p_rayleigh_peak,
            'mean_ang_dist_days': round(mean_dist_days, 2) if not np.isnan(mean_dist_days) else np.nan,
            'kappa_dist': round(kappa_dist, 3) if not np.isnan(kappa_dist) else np.nan,
            'rayleigh_p_dist': p_rayleigh_dist,
            'circ_reg_coef_sin': round(circ_reg_coef_sin, 3) if not np.isnan(circ_reg_coef_sin) else np.nan,
            'circ_reg_coef_cos': round(circ_reg_coef_cos, 3) if not np.isnan(circ_reg_coef_cos) else np.nan,
            'circ_reg_r2': round(circ_reg_r2, 3) if not np.isnan(circ_reg_r2) else np.nan,
            'circ_reg_p': circ_reg_p,
        })

    return pd.DataFrame(out), week_df


# ---- Usage ----
results_circ, week_data = LGP_circular_test(all_df, species_list)
print(results_circ.to_string(index=False, float_format='%.3f'))
