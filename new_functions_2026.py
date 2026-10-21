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

#test
res = LGP_logistic(all_df, species_list)
print(res.to_string(index=False, float_format='%.3f'))
