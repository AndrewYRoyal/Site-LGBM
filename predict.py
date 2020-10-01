#!/usr/bin/env python

import json
import pandas as pd
import lightgbm as lgb
import utility
import os
import time
from datetime import timedelta
import argparse
import boto3
import subprocess

parser = argparse.ArgumentParser('Run LightGBM model and predictions')
parser.add_argument('--log', dest='log', default=False, action='store_true')
args = parser.parse_args()


print('Importing data and parameters')
data_path = 'input/predict_input.csv'
param_path = 'input/model_params.json'

if(not os.path.exists('input')):
    os.mkdir('input')

try:
    s3 = boto3.client('s3')
    s3.download_file('lightgbm-input', 'predict_input.csv', data_path)
    s3.download_file('lightgbm-input', 'model_params.json', param_path)
    print('Input data imported successfully.')
except:
    sys.exit('Error: Input data not found.')

# Import and format data and parameters
#============================================
input_dat = pd.read_csv(data_path)
with open(param_path) as f:
    model_params = json.load(f)

# Unlist selected parameters
for k, v in model_params.items():
    params = v['params']
    params['scoring'], params['verbose'], params['n_jobs'], params['cv'], params['estimator'] = \
        [params[x][0] for x in ['scoring', 'verbose', 'n_jobs', 'cv', 'estimator']]
    v['dep_var'] = v['dep_var'][0]
    v['log_dep'] = v['log_dep'][0]

if(args.log):
    for k, v in model_params.items():
        v['meta'].update({'dep_var': v['dep_var'], 'estimator': v['params']['estimator']})

# Format directories
#============================================
if(not os.path.exists('output')):
    os.mkdir('output')
export_dirs = {
    'models': 'output/models/',
    'predictions': 'output/predictions/',
    'importance': 'output/importance/',
    'params': 'output/params/',
    'error': 'output/error/'}
for k, d in export_dirs.items():
    if(not os.path.exists(d)):
        os.mkdir(d)
dep_vars = list(model_params.keys())
print('Dependent Variables:', ', '.join(dep_vars))
export_paths = {x: {k: d + x + '{}' for k, d in export_dirs.items()} for x in dep_vars}

# Estimate and Predict
#============================================
print('Starting estimation...')

startTime = time.process_time()
for k, feature in model_params.items():
    feature.update({'dat': input_dat})
    print('Estimating {} model'.format(k))
    cv_output = utility.feature_cv(**feature)
    predict_output = utility.feature_predict(**cv_output)
    importance_dat = utility.tab_importance(**predict_output)
    predict_output['model'].booster_.save_model(export_paths[k]['models'].format('.txt'))
    predict_output['predictions'].to_csv(export_paths[k]['predictions'].format('.csv'), index=False)
    importance_dat.to_csv(export_paths[k]['importance'].format('.csv'), index=False)
    opt_params = cv_output['model'].best_params_
    params_dat = pd.DataFrame.from_dict({k: [v] for k, v in opt_params.items()}, orient='columns')
    params_dat.to_csv(export_paths[k]['params'].format('.csv'), index=False)
    pd.DataFrame.from_dict(predict_output['error'], orient='index', columns=['value']).to_csv(export_paths[k]['error'].format('.csv'))
    print('Best Params: {}'.format(opt_params))
    print(*predict_output['error'].items(), sep = '\n')
    if(args.log):
        utility.log_model(
            meta = feature['meta'],
            error = predict_output['error'],
            predictions = predict_output['predictions'],
            paths = export_paths[k])
    elapsed_time = time.process_time() - startTime
    print('Time Elapsed: {}'.format(elapsed_time))

# Sync to s3
#============================================
try:
    subprocess.run(['./export_results.sh'])
    print('Results synced to s3.')
except:
    print('Unable to sync results to s3 bucket.')
