#!/usr/bin/env python

import json
import pandas as pd
import lightgbm as lgb
import utility
import os
import time
from datetime import timedelta

# Import Data
#============================================
siteDT = pd.read_csv('input/predict_input.csv')
with open('input/model_params.json') as f:
    model_params = json.load(f)

# Unlist selected parameters
for k, v in model_params.items():
    params = v['params']
    params['scoring'], params['verbose'], params['n_jobs'], params['cv'] = \
        [params[x][0] for x in ['scoring', 'verbose', 'n_jobs', 'cv']]
    v['dep_var'] = v['dep_var'][0]
    v['log_dep'] = v['log_dep'][0]

# Format directories
#============================================
if(not os.path.exists('output')):
    os.mkdir('output')
export_dirs = {'models': 'output/models/', 'predictions': 'output/predictions/', 'importance': 'output/importance/', 'params': 'output/params/'}
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
    feature.update({'dat': siteDT})
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
    print('Best Params: {}'.format(opt_params))
    print('mape: {mape} aape: {aape}'.format(**predict_output['error']))

    elapsed_time = time.process_time() - startTime
    print('Time Elapsed: {}'.format(elapsed_time))
