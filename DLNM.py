import pandas as pd
import numpy as np
import statsmodels.api as sm
from dlnm import crossbasis, logknots

def test_gravity_lag_dlnm(all_df, species_list, max_lag=7, years=None):
    """
    Test for a lagged effect of continuous gravitational force on harvest
    using a Distributed Lag Non-Linear Model (DLNM).

    Hypothesis: A significant lagged response exists between gravitational
    force and harvest volume.

    Parameters
    ----------
    all_df : dict
        {year: {'df': DataFrame}} with columns 'date', 'gravitation pull (N)',
        and species columns.
    species_list : list of str
    max_lag : int
        Maximum number of days to consider for the lag effect.
    years : iterable, optional

    Returns
    -------
    results : dict
        Per-species summary of the model and the cross-prediction object.
    """
    if years is None:
        years = sorted(all_df.keys())

    # ---- 1. Concatenate daily data across all years ----
    daily_frames = []
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        # Keep only what we need for the model
        cols = ['date', 'gravitation pull (N)'] + species_list
        daily_frames.append(df[cols])

    # Sort by date to ensure correct lag structure
    full_df = pd.concat(daily_frames).sort_values('date').reset_index(drop=True)

    # Ensure continuous daily series (fill missing days if necessary)
    full_df = full_df.set_index('date').asfreq('D').reset_index()

    # ---- 2. Build the cross-basis for gravitational force ----
    # Natural cubic spline for the exposure-response (non-linear effect)
    # Natural cubic spline for the lag-response (smooth decay over time)
    # logknots places knots at equal intervals on the log scale, which is
    # standard for the lag dimension [citation:1][citation:5].
    lknots = logknots(max_lag, nk=3)
    cb_gravity = crossbasis(
        full_df['gravitation pull (N)'].values,
        lag=max_lag,
        argvar={'fun': 'ns', 'df': 3},       # non-linear exposure effect
        arglag={'fun': 'ns', 'knots': lknots} # smooth lag effect
    )

    # ---- 3. Fit a Poisson GLM for each species ----
    results = {}
    for sp in species_list:
        if sp not in full_df.columns:
            continue

        y = full_df[sp].fillna(0).values  # treat missing as zero harvest

        # Build design matrix: intercept + cross-basis + time trend
        # A simple linear time trend helps absorb long-term drift.
        time_idx = np.arange(len(full_df))
        X = np.column_stack([np.ones(len(full_df)), cb_gravity, time_idx])

        try:
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

            # The cross-basis coefficients are positions 1 through n_cb
            n_cb = cb_gravity.shape[1]
            cb_coef = model.params[1:n_cb + 1]
            cb_vcov = model.cov_params().values[1:n_cb + 1, 1:n_cb + 1]

            # Use crosspred to get the full exposure-lag-response surface
            # (requires the dlnm package's prediction utilities)
            from dlnm import crosspred
            pred = crosspred(
                cb_gravity, coef=cb_coef, vcov=cb_vcov,
                model_link='log', at=np.linspace(
                    full_df['gravitation pull (N)'].min(),
                    full_df['gravitation pull (N)'].max(),
                    10
                )
            )

            results[sp] = {
                'model': model,
                'crosspred': pred,
                'n_obs': len(y),
                'max_lag': max_lag,
                'aic': model.aic,
            }
        except Exception as e:
            results[sp] = {'error': str(e)}

    return results
