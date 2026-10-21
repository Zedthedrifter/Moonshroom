#!/usr/bin/env python3
# 2025.08.12
# Zed ZChen@rbge.org.uk
# Input: vcf file, with GT:DP available (default)
# Output: csv file with genotype frequency for each species

import pandas as pd
import subprocess as sbp
import sys
import argparse
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
import matplotlib.pyplot as plt

"""
not used in the latest analysis at all. 
we didn't use the batch information afterall
"""

# Nested functions

# Find gravitational peaks
def has_gravity_peak(week_group): #or nadir
  # Sort by date to ensure chronological orderall_df
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

# find peaked day
def Harvest_peak_day(week_group,harvest):
  # Sort by date to ensure chronological order
  week_group = week_group.sort_values('date')
  Harvest = list(week_group[harvest].values)
  Days = list(week_group['Days'].values)
  collect=[]
  for h,d in zip(Harvest,Days):
    if h==max(Harvest):
      collect.append(d[:3])
  return (','.join(collect))
# If peaks on Friday
def if_Peak_Friday(week_group,harvest):
  # Sort by date to ensure chronological order
  week_group = week_group.sort_values('date')
  Harvest = list(week_group[harvest].values)
  Days = list(week_group['Days'].values)
  collect,ifF=[],''
  for h,d in zip(Harvest,Days):
    if h==max(Harvest):
      collect.append(d)
  if 'Friday' in collect:
    ifF = True
  else:
    ifF = False
  return (ifF)

# Load data
def process_one_year(csv,year,batches,mushroom_columns):
  df = pd.read_csv(csv)
  if 'date ' in df.columns:
    df['date']=df['date '] #deal with a wrong label
  df['date'] = pd.to_datetime(df['date'])
  
  # Add day of week (0=Monday... 6=Sunday)
  df['dayofweek'] = df['date'].dt.dayofweek  # Friday = 4
  
  # Create new dataframe with these columns
  mushroom_df = df[mushroom_columns].copy()
  meta = df.drop(columns=mushroom_columns).copy()
  return (meta,mushroom_df)

#ANALYSE ONE TYPE OF MUSHROOM
def process_one_batch(meta,mushroom_df,batch,harvest,members=3):
  print(f"PROCESSING {harvest} HARVEST")
  df=meta.copy()
  df[batch]=mushroom_df[batch]
  df[harvest]=mushroom_df[harvest]
  df = df.dropna(subset=[batch])
  df[harvest].fillna(0, inplace=True)  
  # Identify weeks with peak gravitational pulls
  gravity_peaks = df.groupby(batch).apply(has_gravity_peak,include_groups=False)
  gravity_peaks = gravity_peaks.rename('has_gravity_peak')
  df = df.merge(gravity_peaks, left_on=batch, right_index=True)
  # Identify weeks with full moons
  fullmoon = df.groupby(batch).apply(has_fullmoon,include_groups=False)
  fullmoon = fullmoon.rename('has_full_moon')
  df = df.merge(fullmoon, left_on=batch, right_index=True)
  # For each batch, compute total harvest
  weekly_totals = df.groupby(batch)[harvest].sum().rename('week_total')  
  df = df.merge(weekly_totals, left_on=batch, right_index=True)
  # Filter groups with at least 3 rows
  df = df.groupby(batch).filter(lambda x: len(x) >= members)
  # Identify weeks with peak harvest on Friday
  peaks = df.groupby(batch).apply(lambda x: Harvest_peak_day(x,harvest),include_groups=False)
  peakF = df.groupby(batch).apply(lambda x: if_Peak_Friday(x,harvest),include_groups=False)
  peaks = peaks.rename('peak_day')
  peakF = peakF.rename('peak_on_Friday')
  df = df.merge(peaks, left_on=batch, right_index=True)
  df = df.merge(peakF, left_on=batch, right_index=True)
  print(df.to_string())
  #merge week info
  week=pd.DataFrame({batch : gravity_peaks.index, 'has_gravity_peak':gravity_peaks.values})
  week=week.merge(fullmoon, left_on=batch, right_index=True)
  week=week.merge(peaks, left_on=batch, right_index=True)
  week=week.merge(peakF, left_on=batch, right_index=True)
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


#Xsquare test
def xsquare(mushroom,year,df,c1,c2):
  f=open(f"chi_results.txt", "a") 
  obv1 = len(df[(df[c1] == True) & (df[c2] == True)])
  obv2 = len(df[(df[c1] == True) & (df[c2] != True)])
  obv3 = len(df[(df[c1] != True) & (df[c2] == True)])
  obv4 = len(df[(df[c1] != True) & (df[c2] != True)])
  fobv=[obv1,obv2,obv3,obv4]
  all=sum(fobv)
  exp1=(obv1+obv3)*(obv1+obv2)/all
  exp2=(obv2+obv4)*(obv1+obv2)/all
  exp3=(obv1+obv3)*(obv3+obv4)/all
  exp4=(obv2+obv4)*(obv3+obv4)/all
  fexp=[exp1,exp2,exp3,exp4]
  chi=chisquare(fobv,fexp)
  f.write(f"{year}{mushroom},{fobv},{fexp},{chi}\n")
  print(f"{mushroom} {year},{fobv},{fexp} {chi}\n")
  return(obv1,obv2,obv2,obv4)

def main(csv, year):
  #print(vcf,names,output)
  batches={'Batch no PO':	'Pleurotus ostreatus', 'Batch no PP':'Pleurotus pulmonarius',	'Batch no Psc':	'Pleurotus saja-caju',	'Batch no Ps':'Pleurotus salmonicolor'}
  mushroom_columns = ['Harvest',
    'Batch no PO', 'Pleurotus ostreatus',
    'Batch no PP', 'Pleurotus pulmonarius', 
    'Batch no Psc', 'Pleurotus saja-caju',
    'Batch no Ps', 'Pleurotus salmonicolor']
  meta,mushroom_df=process_one_year(csv,year,batches,mushroom_columns)
  X={'obv1':0,'obv2':0,'obv3':0,'obv4':0}
  for batch,harvest in batches.items():
    df,week=process_one_batch(meta,mushroom_df,batch,harvest)
    obv1,obv2,obv3,obv4=xsquare(harvest,year,week,'has_gravity_peak','peak_on_Friday')
    X['obv1']+=obv1
    X['obv2']+=obv2
    X['obv3']+=obv3
    X['obv4']+=obv4
  #DO A COMBINED XSQUARE
  f=open(f"chi_results.txt", "a") 
  obv1=X['obv1']
  obv2=X['obv2']
  obv3=X['obv3']
  obv4=X['obv4']
  fobv=[obv1,obv2,obv3,obv4]
  all=sum(fobv)
  exp1=(obv1+obv3)*(obv1+obv2)/all
  exp2=(obv2+obv4)*(obv1+obv2)/all
  exp3=(obv1+obv3)*(obv3+obv4)/all
  exp4=(obv2+obv4)*(obv3+obv4)/all
  fexp=[exp1,exp2,exp3,exp4]
  chi=chisquare(fobv,fexp)
  f.write(f"{year}{fobv},{fexp},{chi}\n")
  print(f"{year},{fobv},{fexp} {chi}\n")

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
  
