#!/usr/bin/env python3
# 2025.08.12
# Zed ZChen@rbge.org.uk
# Input: vcf file, with GT:DP available (default)
# Output: csv file with genotype frequency for each species

import pandas as pd
import numpy as np
import subprocess as sbp
import sys
import argparse
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
import matplotlib.pyplot as plt

# Nested functions

# Find gravitational peaks
def has_gravity_peak(week_group): #or nadir
  # Sort by date to ensure chronological order
  week_group = week_group.sort_values('date')
  
  # Check if deldot changed from negative to positive
  # We'll look at consecutive pairs of days within the week
  deldot_values = week_group['deldot (km/s)'].values
  for i in range(len(deldot_values) - 1):
      if deldot_values[i] < 0 and deldot_values[i + 1] > 0: #change from negative to positive
        return True
      elif deldot_values[i] > 0 and deldot_values[i + 1] < 0 :
        return False 
  return False

# Find full moon
def has_fullmoon(week_group):
  # Sort by date to ensure chronological order
  week_group = week_group.sort_values('date')
  fraction = week_group['fraction'].values
  if max(fraction) > 99:# moon fraction > 99%
    return True
  else:
    return False

# If peaks on Friday
def Harvest_peak(week_group):
  # Sort by date to ensure chronological order
  week_group = week_group.sort_values('date')
  Harvest = list(week_group['Harvest'].values)
  Friday=Harvest[2]
  #sort by ascending order
  Harvest.sort()
  if max(Harvest) == Friday :# maximum harvest on Friday
    return True
  elif Harvest[-2] == Friday: #second highest harvest on Friday?
    return False
  else:
    return False

def Harvest_peak_day(week_group):
  # Sort by date to ensure chronological order
  week_group = week_group.sort_values('date')
  Harvest = list(week_group['Harvest'].values)
  Days = list(week_group['dayofweek'].values)
  HP=[]
  for h,d in zip(Harvest,Days):
    if h==max(Harvest):
      HP.append(d)
  return(HP)

# Load data
def process_one_year(csv,year):
  df = pd.read_csv(csv)
  if 'date ' in df.columns:
    df['date']=df['date '] #deal with a wrong label
  df['date'] = pd.to_datetime(df['date'])
  
  # Add day of week (0=Monday... 6=Sunday)
  df['dayofweek'] = df['date'].dt.dayofweek  # Friday = 4
  
  # Group into weeks starting Wednesday
  # A common approach: shift dates so that Wed=0, then compute week number
  df['week_start_wed'] = (df['date'] - pd.Timedelta(days=2)).dt.isocalendar().week #week1, week2, etc. 
  
  # For each week, compute total harvest
  weekly_totals = df.groupby('week_start_wed')['Harvest'].sum().rename('week_total')  
  # Merge back
  df = df.merge(weekly_totals, left_on='week_start_wed', right_index=True)
  
  # Remove weeks without harvest
  df['Harvest'].fillna(0, inplace=True)
  df['week_total'].fillna(0, inplace=True)
  df=df[df['week_total']!=0] 
  #print(df.to_string())
  
  # Identify weeks with peak gravitational pulls
  gravity_peaks = df.groupby('week_start_wed').apply(has_gravity_peak)
  gravity_peaks = gravity_peaks.rename('has_gravity_peak')
  df = df.merge(gravity_peaks, left_on='week_start_wed', right_index=True)
  #print(df.to_string())
  
  # Identify weeks with full moons
  fullmoon = df.groupby('week_start_wed').apply(has_fullmoon)
  fullmoon = fullmoon.rename('has_full_moon')
  df = df.merge(fullmoon, left_on='week_start_wed', right_index=True)
  print(df.to_string())
  
  # Identify weeks with peak harvest on Friday
  peakF = df.groupby('week_start_wed').apply(Harvest_peak)
  peakW = df.groupby('week_start_wed').apply(Harvest_peak_day)
  peakF = peakF.rename('peak_on_Friday')
  peakW = peakW.rename('Harvest_Peak')
  df = df.merge(peakF, left_on='week_start_wed', right_index=True)
  df = df.merge(peakW, left_on='week_start_wed', right_index=True)
  
  #merge week info
  week=pd.DataFrame({'week_start_wed':gravity_peaks.index, 'has_gravity_peak':gravity_peaks.values})
  week=week.merge(fullmoon, left_on='week_start_wed', right_index=True)
  week=week.merge(peakF, left_on='week_start_wed', right_index=True)
  #week=week.merge(peakW, left_on='week_start_wed', right_index=True)
  def output_zone(peakW,year):
    plot_harvest_day(peakW,year)
    print(df.to_string())
    print(week.to_string())
    df.to_csv(f'{year}_summary.csv', index=True) 
    week.to_csv(f'{year}_week_summary.csv', index=True) 
  #output_zone(peakW,year)
  #print(week.to_string())
  return (df,week)

#Harvest peak day summary
def plot_harvest_day(days,year):
  day_dict={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
  days=list(days)
  x=day_dict.values()
  y=[days.count(i) for i in day_dict]
  print(x,y)
  f = plt.figure()
  plt.bar(x,y)
  f.savefig(f"peak_harvest_day_{year}.pdf", bbox_inches='tight')
  #plt.show()

# added 2026, a bit messy
def dotplot_harvest(df):
  def map_day(d):
    # Shift so that Wednesday becomes 8
    return ((d - 2) % 7) + 8
  df['day_scale'] = df['dayofweek'].apply(map_day)
  plt.figure(figsize=(8, 5))
  plt.scatter(df['day_scale'], df['Harvest'], s=50, c='darkgreen')
  plt.xlabel('Day of week (Wed=8, Thu=9, …, Tue=14)')
  plt.ylabel('Harvest')
  plt.title('Harvest vs Day of Week')
  plt.xticks(range(8, 15))
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.show()

def yield_curve_per_species(df):
  df['date'] = pd.to_datetime(df['date'])
  # Your existing 'dayofweek' uses Monday=0, Sunday=6.
  # We want a cycle that starts on Wednesday.
  # Shift so that Wednesday becomes day 0 of the cycle.
  # dayofweek: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
  # cycle_offset = (dayofweek - 2) % 7   -> Wed=0, Thu=1, …, Tue=6
  df['cycle_offset'] = (df['dayofweek'] - 2) % 7
  # Compute the cycle start (the most recent Wednesday on or before each date)
  # If cycle_offset == 0, the date itself is Wednesday.
  df['cycle_start'] = df['date'] - pd.to_timedelta(df['cycle_offset'], unit='D')
  # Use cycle_start as the grouping key for each fruiting cycle
  cycle_groups = df.groupby('cycle_start')
  # ---------- 2. Species to plot ----------
  species_list = [
      'Pleurotus ostreatus',
      'Pleurotus pulmonarius',
      'Pleurotus saja-caju',
      'Pleurotus salmonicolor'
  ]
  # ---------- 3. Plotting ----------
  fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True)
  axes = axes.ravel()
  for ax, sp in zip(axes, species_list):
      # Collect one curve per cycle
      plotted = 0
      for cycle_start, group in cycle_groups:
          # Skip cycles with no data for this species
          if sp not in group.columns:
              continue
          vals = group[['cycle_offset', sp]].dropna()
          if vals.empty:
              continue
          y = vals[sp].values.astype(float)
          x = vals['cycle_offset'].values  # 0 = Wed, 1 = Thu, …, 6 = Tue
          # Normalize to the maximum within this cycle
          ymax = y.max()
          if ymax == 0 or np.isnan(ymax):
              continue
          y_norm = y / ymax
          ax.plot(x, y_norm, marker='o', alpha=0.6,
                  label=str(cycle_start.date()),color='k')
          plotted += 1
      ax.set_title(sp)
      ax.set_xticks(range(7))
      ax.set_xticklabels(['Wed','Thu','Fri','Sat','Sun','Mon','Tue'])
      ax.set_ylabel('Normalized harvest (0–1)')
      ax.grid(True, linestyle='--', alpha=0.4)
      if plotted > 0:
          # Only show legend if there are cycles; keep it compact
          ax.legend(fontsize=6, ncol=2, loc='upper right')
      else:
          ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                  transform=ax.transAxes)
  fig.suptitle('Normalized fruiting curves per cycle (Wednesday → Tuesday)',
              fontsize=14)
  plt.tight_layout(rect=[0, 0, 1, 0.96])
  plt.show()


#count occurranc of harvest peaks on each day of the week
def count_HP(df):
  # ---------- 1. Prepare date / cycle information ----------
  df['date'] = pd.to_datetime(df['date'])
  # dayofweek: Mon=0 … Sun=6. Shift so that Wednesday = 0 … Tuesday = 6
  df['cycle_offset'] = (df['dayofweek'] - 2) % 7
  # Start of each fruiting cycle (the most recent Wednesday on or before the date)
  df['cycle_start'] = df['date'] - pd.to_timedelta(df['cycle_offset'], unit='D')
  # Desired display labels for cycle days: Wed=8, Thu=9, …, Tue=14
  day_labels = {0:'Wed', 1:'Thu', 2:'Fri', 3:'Sat', 4:'Sun', 5:'Mon', 6:'Tue'}
  day_numeric = {0:8, 1:9, 2:10, 3:11, 4:12, 5:13, 6:14}
  # ---------- 2. Species list ----------
  species_list = [
      'Pleurotus ostreatus',
      'Pleurotus pulmonarius',
      'Pleurotus saja-caju',
      'Pleurotus salmonicolor'
  ]
  # ---------- 3. Find the day of max yield per cycle per species ----------
  # We'll store results in a dict: {species: [count_wed, count_thu, ..., count_tue]}
  counts = {sp: np.zeros(7, dtype=int) for sp in species_list}
  for sp in species_list:
      if sp not in df.columns:
          continue
      # Work with only rows that have a value for this species
      sub = df[['cycle_start', 'cycle_offset', sp]].dropna(subset=[sp])
      if sub.empty:
          continue
      # For each cycle, find the row with the maximum yield
      for cycle, group in sub.groupby('cycle_start'):
        m = group[sp].max()
        for off in group.loc[group[sp] == m, 'cycle_offset']:
            counts[sp][off] += 1
  # ---------- 4. Plot grouped bar chart ----------
  x = np.arange(7)              # positions for Wed..Tue
  width = 0.2                   # bar width
  fig, ax = plt.subplots(figsize=(10, 6))
  for i, sp in enumerate(species_list):
      ax.bar(x + (i - 1.5) * width, counts[sp], width, label=sp)
  # x-axis: show both the cycle day (Wed..Tue) and your 8..14 scale
  ax.set_xticks(x)
  ax.set_xticklabels([f"{day_labels[i]}\n({day_numeric[i]})" for i in range(7)])
  ax.set_xlabel('Day of maximum yield within cycle')
  ax.set_ylabel('Number of cycles')
  #ax.set_title('Day of maximum yield per fruiting cycle')
  ax.legend(title='Species', bbox_to_anchor=(1.02, 1), loc='upper left')
  ax.grid(axis='y', linestyle='--', alpha=0.4)
  plt.tight_layout()
  plt.show()

#Xsquare test
def xsquare(year,df,c1,c2):
  f=open("chi_results.txt", "a") 
  obv1=len(df[df[c1]==True][df[c2]==True])
  obv2=len(df[df[c1]==True][df[c2]!=True])
  obv3=len(df[df[c1]!=True][df[c2]==True])
  obv4=len(df[df[c1]!=True][df[c2]!=True])
  fobv=[obv1,obv2,obv3,obv4]
  all=sum(fobv)
  exp1=(obv1+obv3)*(obv1+obv2)/all
  exp2=(obv2+obv4)*(obv1+obv2)/all
  exp3=(obv1+obv3)*(obv3+obv4)/all
  exp4=(obv2+obv4)*(obv3+obv4)/all
  fexp=[exp1,exp2,exp3,exp4]
  f.write(f"{year},{fobv},{fexp}\n")
  chi=chisquare(fobv,fexp)
  f.write(f"{year},{chi}\n")
  print(f"{year},{fobv},{fexp}\n")

def main(csv, year):
  #print(vcf,names,output)
  df,week=process_one_year(csv,year)
  #xsquare(year,week,'has_gravity_peak','peak_on_Friday')
  #do it in for loop in terminal:
  #for i in $(ls $dir); do  b=${i/Oysterharvest_/}; c=${b/.csv/}; ./moonshroom_stats.py --csv $dir/$i --year $c; done
## files
if __name__ == '__main__':
  parser = argparse.ArgumentParser(
	  description='calculate GT freq of each species in a vcf file', \
	  usage = 'calculate_snp.freq.py -v <vcf file> -n <ID_species.txt> -o <output_dir/prefix>')
  parser.add_argument('-v','--csv', help='csv file harvest for one year',metavar='vcf')
  parser.add_argument('-y','--year', help='year',metavar='year')
  #parser.add_argument('-n','--names', help='sample ID to species names corresponding file, csv: sample ID/file path, species',metavar='names')
  #parser.add_argument('-i','--high', help='GT freq of species specific GT to be at least xx% in the present species',metavar='high')
  #parser.add_argument('-l','--low', help='GT freq of species specific GT to be at most xx% in the absent species',metavar='low')
  #parser.add_argument('-o','--output',help='path to output directory',metavar='output')
  options = parser.parse_args()
  
  main(options.csv,options.year)
  
