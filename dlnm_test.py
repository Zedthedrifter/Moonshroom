import pandas as pd
import numpy as np
import statsmodels.api as sm
from dlnm import crossbasis, crosspred, logknots, plot_crosspred
from scipy import stats  # ensure scipy.stats is imported at top of file

def run_normalized_dlnm(all_df, species_list, years=None, max_lag=7):
    """
    Fit a DLNM for each species using within-week normalized harvest
    (proportion of weekly maximum) as the outcome.
    """
    if years is None:
        years = sorted(all_df.keys())
    # ---- 1. Build combined daily dataframe with normalized outcome ----
    frames = []
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = yr
        # Wednesday = 0 … Tuesday = 6
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        df['cycle_start'] = df['date'] - pd.to_timedelta(df['cycle_offset'], unit='D')
        frames.append(df)
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values('date').reset_index(drop=True)
    # ---- 2. Normalize each species' harvest within each fruiting cycle ----
    for sp in species_list:
        if sp not in daily.columns:
            continue
        # Within each cycle, divide by the cycle's maximum (0 if max is 0)
        daily[f'{sp}_norm'] = daily.groupby('cycle_start')[sp].transform(
            lambda x: x / x.max() if x.max() > 0 else 0.0
        )
    # ---- 3. DLNM per species ----
    results = {}
    for sp in species_list:
        norm_col = f'{sp}_norm'
        if norm_col not in daily.columns:
            print(f"{sp}: column not found — skipped")
            continue
        # Subset to rows with valid exposure and outcome
        sub = daily[['date', 'gravitation pull (N)', norm_col, 'year']].dropna()
        if len(sub) < 30:
            print(f"{sp}: too few observations ({len(sub)}) — skipped")
            continue
        y = sub[norm_col].values
        # Clip to (0, 1) for quasi-binomial stability
        y = np.clip(y, 1e-6, 1 - 1e-6)
        exposure = sub['gravitation pull (N)'].values
        # ---- 3a. Build cross-basis ----
        # Exposure: natural cubic spline (df=4) to capture non-linearity
        # Lag: natural cubic spline over 0–max_lag days (df=3)
        lag_knots = logknots(max_lag, nk=2)
        cb = crossbasis(
            exposure,
            lag=max_lag,
            argvar={'fun': 'ns', 'df': 4},
            arglag={'fun': 'ns', 'knots': lag_knots},
        )
        # ---- 3b. Fit GLM (quasi-binomial for proportion outcome) ----
        # ---- Drop rows where the cross-basis has NaN (first max_lag rows) ----
        valid = ~np.isnan(cb).any(axis=1)
        cb = cb[valid]
        y = y[valid]
        sub = sub[valid].reset_index(drop=True)

        # Year dummies (after dropping rows, so lengths align)
        year_dummies = pd.get_dummies(sub['year'].astype(int), drop_first=True).values.astype(float)

        # Design matrix
        X = np.column_stack([np.ones(len(sub)), cb, year_dummies])

        try:
            model = sm.GLM(y, X, family=sm.families.Binomial()).fit(scale='X2')
        except Exception as e:
            print(f"{sp}: model fit failed — {e}")
            print(f"{sp}: NaN in outcome: {np.isnan(y).sum()}")
            print(f"{sp}: NaN in exposure: {np.isnan(exposure).sum()}")
            print(f"{sp}: NaN in cb: {np.isnan(cb).sum()}")
            print(f"{sp}: Inf in cb: {np.isinf(cb).sum()}")
            continue
        # ---- 3c. Predict lag-response at high gravitational force (LGP) ----
        # Use the 95th percentile of exposure as the "LGP" reference
        lgp_force = np.percentile(exposure, 95)
        pred = crosspred(
            cb,
            coef=model.params[1:cb.shape[1] + 1],
            vcov=model.cov_params()[1:cb.shape[1] + 1, 1:cb.shape[1] + 1],
            at=lgp_force,
            model_link='logit',
            cen=np.median(exposure),
        )
        results[sp] = {
            'model': model,
            'cb': cb,
            'pred': pred,
            'lgp_force': lgp_force,
            'n_obs': len(sub),
        }
    # ---- 3d. Lag-specific ORs, 95% CIs, and Wald p-values ----
        # pred.matfit : fitted values on the logit scale, shape (1, max_lag+1)
        # pred.matse   : standard errors on the logit scale
        log_or = pred.matfit[0, :]            # log odds ratios per lag
        se     = pred.matse[0, :]             # standard errors per lag
        z      = log_or / se
        p_vals = 2 * (1 - stats.norm.cdf(np.abs(z)))

        lag_table = pd.DataFrame({
            'lag':    np.arange(max_lag + 1),
            'OR':     np.exp(log_or),
            'CI_low': np.exp(log_or - 1.96 * se),
            'CI_high':np.exp(log_or + 1.96 * se),
            'p_value':p_vals,
        })

        # ---- 3e. Overall likelihood-ratio test for the cross-basis ----
        # Null model: intercept + year dummies only (no cross-basis)
        X_null = np.column_stack([np.ones(len(sub)), year_dummies])
        model_null = sm.GLM(y, X_null, family=sm.families.Binomial()).fit(scale='X2')
        lr_stat = 2 * (model.llf - model_null.llf)
        df_diff = cb.shape[1]
        lr_p = stats.chi2.sf(lr_stat, df_diff)

        print(f"\n=== {sp} ===")
        print(f"  Observations: {len(sub)}")
        print(f"  LGP force (95th pct): {lgp_force:.3e} N")
        print(lag_table.to_string(index=False, float_format='%.3f'))
        print(f"  Overall LR test for cross-basis: chi2 = {lr_stat:.2f}, "
                f"df = {df_diff}, p = {lr_p:.4f}")
    return results, daily

#USAGE
results, daily=run_normalized_dlnm(all_df, species_list, years=None, max_lag=7)