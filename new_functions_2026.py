import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

species_list = [
      'Pleurotus ostreatus',
      'Pleurotus pulmonarius',
      'Pleurotus saja-caju',
      'Pleurotus salmonicolor'
  ]

# Final result: {species: {year: [list of cycle days]}}
def count_HP(all_df):
    result = {sp: {} for sp in species_list}
    for year, year_dict in all_df.items():
        df = year_dict['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        # Wednesday = 0, Thursday = 1, …, Tuesday = 6
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        df['cycle_start'] = df['date'] - pd.to_timedelta(df['cycle_offset'], unit='D')
        for sp in species_list:
            if sp not in df.columns:
                result[sp][year] = {}
                continue
            sub = df[['cycle_start', 'cycle_offset', sp]].dropna(subset=[sp])
            if sub.empty:
                result[sp][year] = {}
                continue
            # Aggregate multiple records on the same day within a cycle (mean)
            daily = (sub.groupby(['cycle_start', 'cycle_offset'])[sp]
                        .mean().reset_index())
            peak_dict = {}
            for cycle, group in daily.groupby('cycle_start'):
                non_zero = group[group[sp] > 0]
                if len(non_zero) < 3:
                    continue  # require ≥3 non-zero days in this cycle
                m = non_zero[sp].max()
                tied_days = non_zero.loc[non_zero[sp] == m, 'cycle_offset'].astype(int).tolist()
                # If there is a tie, choose the first occurrence (smallest offset = earliest day)
                peak_day = list(tied_days)
                # Store as {week_start_date: peak_day}
                peak_dict[cycle.date()] = peak_day
            result[sp][year] = peak_dict
    return result

#PLOT THE HP FREQUENCY DISTRIBUTION

def HP_freq_distribution(HPs):
    years = list(range(2004, 2011))  # 2004–2010 inclusive
    # result structure: {species: {year: {week_start_date: peak_day(0–6)}}}
    # Step 1: for each species, count how many years had a peak on each cycle day
    counts = {}  # {species: {day: [counts per year]}}
    for sp in species_list:
        counts[sp] = {}
        for day in range(7):  # 0=Wed … 6=Tue
            counts[sp][day] = []
        for yr in years:
            peak_dict = HPs.get(sp, {}).get(yr, {})
            # count how many cycles in this year had peak on each day
            year_counts = np.zeros(7, dtype=int)
            for wk, day in peak_dict.items():
                year_counts[day] += 1
            for day in range(7):
                counts[sp][day].append(year_counts[day])
    # Step 2: build the boxplot
    day_labels = ['Wed\n(8)', 'Thu\n(9)', 'Fri\n(10)', 'Sat\n(11)', 'Sun\n(12)', 'Mon\n(13)', 'Tue\n(14)']
    colors = {'Pleurotus ostreatus': '#4C72B0',
            'Pleurotus pulmonarius': '#DD8452',
            'Pleurotus saja-caju': '#55A868'}
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    x = np.arange(7)
    for i, sp in enumerate(species_list):
        positions = x + (i - 1) * width
        data = [counts[sp][d] for d in range(7)]
        bp = ax.boxplot(data, positions=positions, widths=width*0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(facecolor=colors[sp], alpha=0.5),
                        whiskerprops=dict(color=colors[sp]),
                        capprops=dict(color=colors[sp]))
        # overlay individual points (7 years)
        for d in range(7):
            y_vals = counts[sp][d]
            jitter = np.random.uniform(-width*0.15, width*0.15, size=len(y_vals))
            ax.scatter(positions[d] + jitter, y_vals, s=30,
                    color=colors[sp], edgecolor='black', linewidth=0.4,
                    alpha=0.9, zorder=3, label=sp if d == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel('Day from fruiting initiation (Wed=8 … Tue=14)')
    ax.set_ylabel('Count of peak harvest days (HP) per year')
    ax.set_title('Peak harvest day distribution per species (2004–2010)')
    ax.legend(title='Species', loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

def HP_freq_stats(HPs, species_list, years=range(2004, 2011)):
    """
    Compute descriptive statistics of peak-harvest (HP) counts.
    Parameters
    ----------
    HPs : dict
        {species: {year: {week_start_date: peak_day(0–6)}}}
    species_list : list of str
        Species names (columns in the original data).
    years : iterable of int
        Years to include (default 2004–2010).
    Returns
    -------
    pd.DataFrame
        Columns: species, day, day_label, n, mean, sd, se, median, min, max
    """
    day_labels = ['Wed (8)', 'Thu (9)', 'Fri (10)', 'Sat (11)',
                  'Sun (12)', 'Mon (13)', 'Tue (14)']
    rows = []
    for sp in species_list:
        # counts[day] = list of per-year counts (one per year in `years`)
        counts = {d: [] for d in range(7)}
        for yr in years:
            peak_dict = HPs.get(sp, {}).get(yr, {})
            year_counts = np.zeros(7, dtype=int)
            for wk, day in peak_dict.items():
                year_counts[day] += 1
            for d in range(7):
                counts[d].append(year_counts[d])
        for d in range(7):
            vals = np.array(counts[d], dtype=float)
            n = len(vals)
            mean = vals.mean() if n else np.nan
            sd = vals.std(ddof=1) if n > 1 else 0.0
            se = sd / np.sqrt(n) if n > 1 else 0.0
            rows.append({
                'species': sp,
                'day': d,
                'day_label': day_labels[d],
                'n': n,
                'mean': mean,
                'sd': sd,
                'se': se,
                'median': np.median(vals) if n else np.nan,
                'min': vals.min() if n else np.nan,
                'max': vals.max() if n else np.nan,
            })
    return pd.DataFrame(rows)

def HP_freq_table(HPs, species_list, years=range(2004, 2011)):
    """
    Build a descriptive-statistics table of peak-harvest (HP) counts.
    Returns a pandas DataFrame with:
      - index   : species
      - columns : 8, 9, 10, 11, 12, 13, 14  (Wed … Tue)
      - values  : "mean|median (se)" strings
    """
    # Map cycle day (0–6) to your numeric scale (8–14)
    day_to_num = {d: d + 8 for d in range(7)}
    rows = {}
    for sp in species_list:
        # counts[day] = list of per-year counts
        counts = {d: [] for d in range(7)}
        for yr in years:
            peak_dict = HPs.get(sp, {}).get(yr, {})
            year_counts = np.zeros(7, dtype=int)
            for wk, day in peak_dict.items():
                year_counts[day] += 1
            for d in range(7):
                counts[d].append(year_counts[d])
        # Build the row: "mean|median (se)" per day
        row = {}
        for d in range(7):
            vals = np.array(counts[d], dtype=float)
            n = len(vals)
            mean = vals.mean() if n else np.nan
            median = np.median(vals) if n else np.nan
            sd = vals.std(ddof=1) if n > 1 else 0.0
            se = sd / np.sqrt(n) if n > 1 else 0.0
            row[day_to_num[d]] = f"{mean:.2f}|{median:.1f} ({se:.2f})"
        rows[sp] = row
    df = pd.DataFrame(rows).T
    df.columns = [f"{c}" for c in df.columns]  # keep 8, 9, ... 14
    df.index.name = None
    return df

def LGP_logistic(all_df, species_list, years=None):
    """
    Test whether lunar gravitational peaks (LGP) reduce the probability
    of a peak harvest on day 10 (Friday), per species.
    Parameters
    ----------
    all_df : dict
        {year: {'df': DataFrame}} with columns including
        'date', 'dayofweek', 'week_start_wed', 'has_gravity_peak',
        and species columns.
    species_list : list of str
    years : iterable of int, optional
        Years to include. Defaults to all keys in all_df.
    Returns
    -------
    results_df : DataFrame
        Columns: species, n_weeks, n_lgp, OR, CI_low, CI_high,
                 p_two, p_one, n_day10, n_day10_lgp
    """
    if years is None:
        years = sorted(all_df.keys())
    # ---- 1. Build week-level dataframe ----
    rows = []
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        # Wednesday = 0 … Tuesday = 6
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        df['cycle_start'] = df['date'] - pd.to_timedelta(df['cycle_offset'], unit='D')
        # LGP flag per cycle: any day in the cycle with has_gravity_peak == True
        lgp_by_cycle = (
            df.groupby('cycle_start')['has_gravity_peak']
              .any().astype(int).to_dict()
        )
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
                peak_day = min(tied)
                rows.append({
                    'species': sp,
                    'year': yr,
                    'cycle_start': cycle,
                    'peak_day': peak_day,
                    'hp_on_day10': int(peak_day == 2),   # Friday = day 10
                    'lgp': lgp_by_cycle.get(cycle, 0),
                })
    week_df = pd.DataFrame(rows)
    if week_df.empty:
        raise ValueError("No weeks passed the ≥3 non-zero filter.")
    # ---- 2. Fit logistic regression per species ----
    out = []
    for sp in species_list:
        sub = week_df[week_df['species'] == sp].copy()
        if sub.empty:
            continue
        if sub['lgp'].nunique() < 2:
            out.append({
                'species': sp, 'n_weeks': len(sub),
                'n_lgp': int(sub['lgp'].sum()),
                'OR': np.nan, 'CI_low': np.nan, 'CI_high': np.nan,
                'p_two': np.nan, 'p_one': np.nan,
                'n_day10': int(sub['hp_on_day10'].sum()),
                'n_day10_lgp': int(sub.loc[sub['lgp'] == 1, 'hp_on_day10'].sum()),
            })
            continue
        model = smf.logit('hp_on_day10 ~ lgp + C(year)', data=sub).fit(disp=0)
        beta = model.params['lgp']
        or_ = np.exp(beta)
        ci = model.conf_int().loc['lgp']
        ci_lo, ci_hi = np.exp(ci[0]), np.exp(ci[1])
        p_two = model.pvalues['lgp']
        p_one = p_two / 2 if beta < 0 else 1 - p_two / 2
        out.append({
            'species': sp,
            'n_weeks': len(sub),
            'n_lgp': int(sub['lgp'].sum()),
            'OR': or_,
            'CI_low': ci_lo,
            'CI_high': ci_hi,
            'p_two': p_two,
            'p_one': p_one,
            'n_day10': int(sub['hp_on_day10'].sum()),
            'n_day10_lgp': int(sub.loc[sub['lgp'] == 1, 'hp_on_day10'].sum()),
        })
    return pd.DataFrame(out)

def LGP_freq_distribution(all_df, species_list, years=None, percentile=95):
    """
    Boxplot of the number of high-gravitational-pull days per cycle day,
    with one point per year (same format as HP_freq_distribution).
    A "high gravitational pull day" is any day whose 'gravitation pull (N)'
    is in the top (100 - percentile) percent of all days in the dataset.
    """
    if years is None:
        years = sorted(all_df.keys())

    # ---- 1. Determine the global high-force threshold ----
    all_force = []
    for yr in years:
        df = all_df[yr]['df']
        all_force.append(df['gravitation pull (N)'].dropna().values)
    all_force = np.concatenate(all_force)
    threshold = np.percentile(all_force, percentile)
    print(f"High-force threshold (top {100 - percentile}%): {threshold:.3e} N")

    # ---- 2. Count high-force days per cycle day per year ----
    # counts[day] = list of yearly counts (one per year)
    counts = {d: [] for d in range(7)}
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7

        # Flag high-force days
        high = df['gravitation pull (N)'] >= threshold

        # Count per cycle day in this year
        year_counts = np.zeros(7, dtype=int)
        for d in range(7):
            year_counts[d] = int(((df['cycle_offset'] == d) & high).sum())
        for d in range(7):
            counts[d].append(year_counts[d])

    # ---- 3. Build the boxplot ----
    day_labels = ['Wed\n(8)', 'Thu\n(9)', 'Fri\n(10)', 'Sat\n(11)',
                  'Sun\n(12)', 'Mon\n(13)', 'Tue\n(14)']
    color = '#8E44AD'  # single colour for this plot

    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.5
    x = np.arange(7)
    data = [counts[d] for d in range(7)]

    bp = ax.boxplot(data, positions=x, widths=width * 0.9,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor=color, alpha=0.5),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color))

    # Overlay individual points (one per year)
    for d in range(7):
        y_vals = counts[d]
        jitter = np.random.uniform(-width * 0.15, width * 0.15,
                                   size=len(y_vals))
        ax.scatter(x[d] + jitter, y_vals, s=30,
                   color=color, edgecolor='black', linewidth=0.4,
                   alpha=0.9, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel('Day from fruiting initiation (Wed=8 … Tue=14)')
    ax.set_ylabel(f'Count of high-force days (top {100 - percentile}%) per year')
    ax.set_title(f'Distribution of high-gravitational-pull days '
                 f'across the week (threshold = {threshold:.2e} N)')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return counts, threshold

def test_uniform_day_distribution(all_df, years=None, percentile=95, n_perm=10000, seed=42):
    """
    Test whether high-gravitational-pull days are uniformly distributed
    across the seven days of the fruiting cycle (Wed=0 … Tue=6).

    Returns
    -------
    dict with:
      - observed_counts : array of length 7 (high-force days per cycle day)
      - chi2_stat, chi2_p : standard chi-square GOF (approximate)
      - perm_p : permutation p-value (valid with small counts)
      - n_high : total number of high-force days
    """
    if years is None:
        years = sorted(all_df.keys())

    # ---- 1. Threshold from all years combined ----
    all_force = np.concatenate([
        all_df[yr]['df']['gravitation pull (N)'].dropna().values
        for yr in years
    ])
    threshold = np.percentile(all_force, percentile)

    # ---- 2. Observed counts per cycle day ----
    observed = np.zeros(7, dtype=int)
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        high = df['gravitation pull (N)'] >= threshold
        for d in range(7):
            observed[d] += int(((df['cycle_offset'] == d) & high).sum())

    n_high = observed.sum()

    # ---- 3. Standard chi-square GOF (uniform expectation) ----
    expected = np.full(7, n_high / 7)
    chi2_stat, chi2_p = stats.chisquare(observed, expected)

    # ---- 4. Permutation test ----
    # Null: cycle day label is independent of the high-force flag.
    # Permute the high-force flags within each year (preserving year sizes).
    rng = np.random.default_rng(seed)
    perm_stats = np.empty(n_perm)

    # Pre-build per-year arrays of (cycle_offset, high_flag)
    year_data = []
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        high = (df['gravitation pull (N)'] >= threshold).values
        offsets = df['cycle_offset'].values
        year_data.append((offsets, high))

    for i in range(n_perm):
        perm_counts = np.zeros(7, dtype=int)
        for offsets, high in year_data:
            shuffled = rng.permutation(high)   # shuffle the high-force flags
            for d in range(7):
                perm_counts[d] += int(((offsets == d) & shuffled).sum())
        # chi-square statistic under permutation (uniform expectation)
        exp = np.full(7, perm_counts.sum() / 7)
        stat = ((perm_counts - exp) ** 2 / exp).sum()
        perm_stats[i] = stat

    perm_p = (perm_stats >= chi2_stat).mean()

    return {
        'observed_counts': observed,
        'chi2_stat': chi2_stat,
        'chi2_p': chi2_p,
        'perm_p': perm_p,
        'n_high': int(n_high),
        'threshold': threshold,
    }

def LGP_and_moon_freq_distribution(all_df, years=None,
                                   gravity_percentile=95,
                                   fraction_percentile=99):
    """
    Boxplot of high-gravity days and high-fraction (full-moon) days
    per cycle day, with one point per year. 
    - High gravity: top (100 - gravity_percentile)% of 'gravitation pull (N)'
    - High fraction (full moon): top (100 - fraction_percentile)% of 'fraction'
      (i.e., 99th percentile cutoff by default)
    Produces one figure with two sets of boxes:
      - purple: high gravitational pull
      - golden: high fraction (full moon)
    """
    if years is None:
        years = sorted(all_df.keys())
    # ---- 1. Global thresholds ----
    all_force = np.concatenate([
        all_df[yr]['df']['gravitation pull (N)'].dropna().values
        for yr in years
    ])
    gravity_threshold = np.percentile(all_force, gravity_percentile)
    all_fraction = np.concatenate([
        all_df[yr]['df']['fraction'].dropna().values
        for yr in years
    ])
    fraction_threshold = np.percentile(all_fraction, fraction_percentile)
    print(f"High-gravity threshold (top {100 - gravity_percentile}%): "
          f"{gravity_threshold:.3e} N")
    print(f"High-fraction threshold (top {100 - fraction_percentile}%): "
          f"{fraction_threshold:.3f}")
    # ---- 2. Count per cycle day per year ----
    gravity_counts = {d: [] for d in range(7)}
    moon_counts    = {d: [] for d in range(7)}
    for yr in years:
        df = all_df[yr]['df'].copy()
        df['date'] = pd.to_datetime(df['date'])
        df['cycle_offset'] = (df['dayofweek'] - 2) % 7
        high_gravity = df['gravitation pull (N)'] >= gravity_threshold
        high_moon    = df['fraction'] >= fraction_threshold
        year_gravity = np.zeros(7, dtype=int)
        year_moon    = np.zeros(7, dtype=int)
        for d in range(7):
            year_gravity[d] = int(((df['cycle_offset'] == d) & high_gravity).sum())
            year_moon[d]    = int(((df['cycle_offset'] == d) & high_moon).sum())
        for d in range(7):
            gravity_counts[d].append(year_gravity[d])
            moon_counts[d].append(year_moon[d])
    # ---- 3. Combined boxplot ----
    day_labels = ['Wed\n(8)', 'Thu\n(9)', 'Fri\n(10)', 'Sat\n(11)',
                  'Sun\n(12)', 'Mon\n(13)', 'Tue\n(14)']
    color_gravity = '#8E44AD'   # purple
    color_moon    = '#D4A017'   # golden
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.35
    x = np.arange(7)
    # Gravity boxes (shifted left)
    gravity_data = [gravity_counts[d] for d in range(7)]
    ax.boxplot(gravity_data,
               positions=x - width/2, widths=width*0.9,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='black', linewidth=1.5),
               boxprops=dict(facecolor=color_gravity, alpha=0.5),
               whiskerprops=dict(color=color_gravity),
               capprops=dict(color=color_gravity))

    # Moon boxes (shifted right)
    moon_data = [moon_counts[d] for d in range(7)]
    ax.boxplot(moon_data,
               positions=x + width/2, widths=width*0.9,
               patch_artist=True, showfliers=False,
               medianprops=dict(color='black', linewidth=1.5),
               boxprops=dict(facecolor=color_moon, alpha=0.5),
               whiskerprops=dict(color=color_moon),
               capprops=dict(color=color_moon))
    # Overlay individual points
    for d in range(7):
        # gravity
        y_vals = gravity_counts[d]
        jitter = np.random.uniform(-width*0.15, width*0.15, size=len(y_vals))
        ax.scatter(x[d] - width/2 + jitter, y_vals, s=28,
                   color=color_gravity, edgecolor='black', linewidth=0.4,
                   alpha=0.9, zorder=3)
        # moon
        y_vals = moon_counts[d]
        jitter = np.random.uniform(-width*0.15, width*0.15, size=len(y_vals))
        ax.scatter(x[d] + width/2 + jitter, y_vals, s=28,
                   color=color_moon, edgecolor='black', linewidth=0.4,alpha=0.9, zorder=3)
    # Legend proxies
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_gravity, alpha=0.5,
              label=f'High gravity (top {100 - gravity_percentile}%)'),
        Patch(facecolor=color_moon, alpha=0.5,
              label=f'Full moon / high fraction (top {100 - fraction_percentile}%)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel('Day from fruiting initiation (Wed=8 … Tue=14)')
    ax.set_ylabel('Count of high-value days per year')
    ax.set_title('Distribution of high-gravitational-pull and full-moon days '
                 'across the week')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    
    return {
        'gravity_counts': gravity_counts,
        'moon_counts': moon_counts,
        'gravity_threshold': gravity_threshold,
        'fraction_threshold': fraction_threshold,
    }


def test_uniform_day_distribution_both(all_df, years=None,
                                       gravity_percentile=95,
                                       fraction_percentile=99,
                                        n_perm=10000, seed=42):
    """
    Test uniformity of high-gravity days and high-fraction (full-moon) days
    across the seven cycle days.
    Returns a dict with two sub-dicts: 'gravity' and 'moon', each containing
    observed_counts, chi2_stat, chi2_p, perm_p, n_high, threshold.
    """
    if years is None:
        years = sorted(all_df.keys())
    # ---- 1. Thresholds ----
    all_force = np.concatenate([
        all_df[yr]['df']['gravitation pull (N)'].dropna().values
        for yr in years
    ])
    gravity_threshold = np.percentile(all_force, gravity_percentile)
    all_fraction = np.concatenate([
        all_df[yr]['df']['fraction'].dropna().values
        for yr in years
    ])
    fraction_threshold = np.percentile(all_fraction, fraction_percentile)
    # ---- 2. Helper for observed counts ----
    def observed_counts(column, threshold):
        obs = np.zeros(7, dtype=int)
        for yr in years:
            df = all_df[yr]['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            df['cycle_offset'] = (df['dayofweek'] - 2) % 7
            high = df[column] >= threshold
            for d in range(7):
                obs[d] += int(((df['cycle_offset'] == d) & high).sum())
        return obs
    # ---- 3. Helper for permutation test ----
    def perm_test(column, threshold):
        year_data = []
        for yr in years:
            df = all_df[yr]['df'].copy()
            df['date'] = pd.to_datetime(df['date'])
            df['cycle_offset'] = (df['dayofweek'] - 2) % 7
            high = (df[column] >= threshold).values
            offsets = df['cycle_offset'].values
            year_data.append((offsets, high))
        rng = np.random.default_rng(seed)
        perm_stats = np.empty(n_perm)
        for i in range(n_perm):
            perm_counts = np.zeros(7, dtype=int)
            for offsets, high in year_data:
                shuffled = rng.permutation(high)
                for d in range(7):
                    perm_counts[d] += int(((offsets == d) & shuffled).sum())
            exp = np.full(7, perm_counts.sum() / 7)
            stat = ((perm_counts - exp) ** 2 / exp).sum()
            perm_stats[i] = stat
        return perm_stats
    # ---- 4. Run for gravity ----
    obs_gravity = observed_counts('gravitation pull (N)', gravity_threshold)
    n_gravity = obs_gravity.sum()
    chi2_g, chi2_p_g = stats.chisquare(obs_gravity, np.full(7, n_gravity/7))
    perm_stats_g = perm_test('gravitation pull (N)', gravity_threshold)
    perm_p_g = (perm_stats_g >= chi2_g).mean()
    # ---- 5. Run for moon ----
    obs_moon = observed_counts('fraction', fraction_threshold)
    n_moon = obs_moon.sum()
    chi2_m, chi2_p_m = stats.chisquare(obs_moon, np.full(7, n_moon/7))
    perm_stats_m = perm_test('fraction', fraction_threshold)
    perm_p_m = (perm_stats_m >= chi2_m).mean()
    return {
        'gravity': {
            'observed_counts': obs_gravity,
            'chi2_stat': chi2_g, 'chi2_p': chi2_p_g,
            'perm_p': perm_p_g,
            'n_high': int(n_gravity),
            'threshold': gravity_threshold,
        },
        'moon': {
            'observed_counts': obs_moon,
            'chi2_stat': chi2_m, 'chi2_p': chi2_p_m,
            'perm_p': perm_p_m,
            'n_high': int(n_moon),
            'threshold': fraction_threshold,
        },
    }
# ---- Usage ----

def main():
    # Species columns of interest
    species_list = [
        'Pleurotus ostreatus',
        'Pleurotus pulmonarius',
        'Pleurotus saja-caju'
    ]

all_df={}
for i in range(2004,2019):
    df,week=process_one_year(f'~/data/Moonshroom/Oysterharvest_{i}.csv',i)
    all_df[i]={'df':df,'week':week}

HPs=count_HP(all_df)
# Display & manual check
for sp in species_list:
    print(f"\n=== {sp} ===")
    for year, peaks in sorted(HPs[sp].items()):
        print(f"  {year}: {peaks}")
HP_freq_distribution(HPs)
HP_freq_stats(HPs, species_list, years=range(2004, 2011))
test_more_HPs_fri(HPs)
table=HP_freq_table(HPs, species_list, years=range(2004, 2011))
counts,threshold=LGP_freq_distribution(all_df, species_list, years=None, percentile=95)

LGP_and_moon_freq_distribution(all_df, years=None,
                                   gravity_percentile=95,
                                   fraction_percentile=99)
test_uniform_day_distribution_both(all_df, years=None,
                                       gravity_percentile=95,
                                       fraction_percentile=99,
                                        n_perm=10000, seed=42)
results=test_uniform_day_distribution(all_df, years=None, percentile=95, n_perm=10000, seed=42)
#test
res = LGP_logistic(all_df, species_list)
print(res.to_string(index=False, float_format='%.3f'))
