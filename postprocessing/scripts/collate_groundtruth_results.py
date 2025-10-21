"""Collates json-formatted results, cleans them up and saves them as .feather
files."""
# Author: William La Cava, williamlacava@gmail.com
# SRBENCH
# License: GPLv3

################################################################################
# Ground-truth problems
################################################################################
import pandas as pd
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import sys
import pdb
from improving_names import *

# Where to load the results
rdir = '../../results_sym_data/'
if len(sys.argv) > 1:
    rdir = sys.argv[1]
else:
    print('no rdir provided, using',rdir)
print('reading results from  directory', rdir)

# Where to save the report
sdir = '../../results/ground-truth/'
if len(sys.argv) > 2:
    sdir = sys.argv[2]
else:
    print('no sdir provided, using', sdir)

print('saving summary to directory', sdir)
if not os.path.exists(sdir):
    print("Creating directory", sdir)
    os.makedirs(sdir)

##########
# load data from json
##########

frames = []
excluded_cols = [
    'params'
]
fails = []
bad_bsr = []
updated = 0
for f in tqdm(glob(rdir + '/*/*.json')):
    if os.path.exists(f+'.updated'):
        f += '.updated'
        updated += 1
    if 'cv_results' in f: 
        continue
    if 'EHC' in f:
        continue

    try: 
        r = json.load(open(f,'r'))
        
        if isinstance(r['symbolic_model'],list):
            print('WARNING: list returned for model:',f)
            bad_bsr.append(f)
            sm = ['B'+str(i)+'*'+ri for i, ri in enumerate(r['symbolic_model'])]
            sm = '+'.join(sm)
            r['symbolic_model'] = sm
            
        sub_r = {k:v for k,v in r.items() if k not in excluded_cols}
    #     df = pd.DataFrame(sub_r)
        frames.append(sub_r) 
    #     print(f)
    #     print(r.keys())
    except Exception as e:
        fails.append([f,e])
        pass
    
print('{} results files loaded, {} ({:.1f}%) of which are '
	'updated'.format(len(frames), updated, updated/len(frames)*100))

print(len(fails),'fails:')
for f in fails: 
    print(f[0])
print('bad bsr:',bad_bsr)

df_results = pd.DataFrame.from_records(frames)

##########
# cleanup
##########
df_results = df_results.rename(columns={'time_time':'training time (s)'})
df_results.loc[:,'training time (hr)'] = df_results['training time (s)']/3600

####################
# Improving names and adding metadata
####################
df_results = improve_names(df_results)
df_results = add_metadata(df_results)

# add modified R2 with 0 floor
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x,0))
for col in ['symbolic_error_is_zero', 'symbolic_error_is_constant',
            'symbolic_fraction_is_constant']:
    df_results.loc[:,col] = df_results[col].fillna(False)
    
print('mean trial count:')
print(df_results.groupby('algorithm')['dataset'].count().sort_values()
      / df_results.dataset.nunique())

##########
# compute symbolic solutions
##########

df_results.loc[:,'symbolic_solution'] = df_results[['symbolic_error_is_zero',
                                                    'symbolic_error_is_constant',
                                                    'symbolic_fraction_is_constant']
                                                   ].apply(any,raw=True, axis=1)

# clean up any corner cases (constant models, failures)
df_results.loc[:,'symbolic_solution'] = \
    df_results['symbolic_solution'] & ~df_results['simplified_symbolic_model'].isna() 
df_results.loc[:,'symbolic_solution'] = \
    df_results['symbolic_solution'] & ~(df_results['simplified_symbolic_model'] == '0')
df_results.loc[:,'symbolic_solution'] = \
    df_results['symbolic_solution'] & ~(df_results['simplified_symbolic_model'] == 'nan')


# Compute average ranks and average number of trials per algorithm
avg_ranks = df_results.groupby('algorithm').agg(
    r2_test_avg_rank=('r2_test_rank', 'mean'),
    model_size_avg_rank=('model_size_rank', 'mean')
).reset_index()

trial_counts = df_results.groupby(['algorithm', 'dataset', 'random_state']).size() \
    .groupby('algorithm').mean().reset_index(name='avg_num_trials')

avg_ranks = avg_ranks.merge(trial_counts, on='algorithm')
avg_ranks = avg_ranks.sort_values('r2_test_avg_rank')

print("\nAverage ranks and average number of trials across datasets:")
print(avg_ranks)


##########
# save results
##########

df_results.to_feather(f'{sdir}/results.feather')
print(f'results saved to {sdir}/results.feather')
