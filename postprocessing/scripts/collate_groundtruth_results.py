"""
Collates JSON-formatted results for ground-truth datasets, cleans them up,
and saves them as .feather files.
"""
# Original Author: William La Cava
# Modified by: Guilherme Aldeia
# License: GPLv3

import os
import sys
import json
from glob import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from improving_names import improve_names, add_metadata

# Directories
rdir = sys.argv[1] if len(sys.argv) > 1 else '../../results_sym_data/'
sdir = sys.argv[2] if len(sys.argv) > 2 else '../../results/ground-truth/'

print('Reading results from directory', rdir)
print('Saving summary to directory', sdir)
os.makedirs(sdir, exist_ok=True)

# Load JSON results
frames, fails, bad_bsr = [], [], []
updated = 0
for f in tqdm(glob(rdir + '/*/*.json')):
    if os.path.exists(f + '.updated'):
        f += '.updated'
        updated += 1
    if 'cv_results' in f or 'EHC' in f:
        continue
    try:
        r = json.load(open(f, 'r'))
        if isinstance(r['symbolic_model'], list):
            print('WARNING: list returned for model:', f)
            bad_bsr.append(f)
            sm = '+'.join([f'B{i}*{ri}' for i, ri in enumerate(r['symbolic_model'])])
            r['symbolic_model'] = sm
        # Remove params column
        r.pop('params', None)
        frames.append(r)
    except Exception as e:
        fails.append([f, e])

print(f'{len(frames)} results files loaded, {updated} ({updated/len(frames)*100:.1f}%) updated')
print(len(fails), 'fails')
for f in fails:
    print(f[0])
print('Bad BSR models:', bad_bsr)

# Build DataFrame
df_results = pd.DataFrame.from_records(frames)

# Cleanup
df_results = df_results.rename(columns={'time_time': 'training time (s)'})
df_results['training time (hr)'] = df_results['training time (s)'] / 3600

# Improve names and add metadata
df_results = improve_names(df_results)
df_results = add_metadata(df_results)

# Additional metrics
df_results['r2_zero_test'] = df_results['r2_test'].apply(lambda x: max(x, 0))
for col in ['symbolic_error_is_zero', 'symbolic_error_is_constant',
            'symbolic_fraction_is_constant']:
    df_results[col] = df_results[col].fillna(False)

# Mean trial count
print('Mean trial count per algorithm:')
print(df_results.groupby('algorithm')['dataset'].count() / df_results.dataset.nunique())

# Compute symbolic solutions
df_results['symbolic_solution'] = df_results[[
    'symbolic_error_is_zero',
    'symbolic_error_is_constant',
    'symbolic_fraction_is_constant'
]].any(axis=1)

# Remove corner cases
df_results['symbolic_solution'] &= df_results['simplified_symbolic_model'].notna()
df_results['symbolic_solution'] &= df_results['simplified_symbolic_model'] != '0'
df_results['symbolic_solution'] &= df_results['simplified_symbolic_model'] != 'nan'

# Compute ranks per dataset for r2_test and model_size
for col in ['r2_test', 'model_size']:
    ascending = col != 'r2_test'  # r2 higher is better
    df_results[col + '_rank'] = df_results.groupby(['dataset', 'random_state'])[col] \
        .transform(lambda x: round(x.rank(ascending=ascending), 3))

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

# Save results
df_results.to_feather(f'{sdir}/results.feather')
print(f'Results saved to {sdir}/results.feather')
