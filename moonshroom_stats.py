#!/home/zchen/apps/env/easy353/bin/python3
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
  for h,d in zip(Harvest,Days):
    if h==max(Harvest):
      return (d)

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
  df = df.merge(peakF, left_on='week_start_wed', right_index=True)
  
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
  print(week.to_string())
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
  xsquare(year,week,'has_gravity_peak','peak_on_Friday')
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
  
