import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

def feature_cv(dat, dep_var, log_dep, ivars, params, train_omit = [''], **kwargs):
    # Filter Data
    dat = dat[['site', dep_var] + ivars]
    dat.rename(columns={dep_var: 'value'}, inplace=True)
    train_dat = dat[dat['value'].notna()]
    train_dat = train_dat[~train_dat['site'].isin(train_omit)]
    train_dat.reset_index(inplace = True, drop = True)
    if (log_dep):
        train_dat['value'] = np.log1p(train_dat['value'])
    # Define grid search
    params.update({'estimator': lgb.LGBMRegressor()})
    gs = GridSearchCV(**params)
    mfit = gs.fit(y = train_dat['value'], X = train_dat.drop(columns = ['site', 'value']))
    return {'model': mfit, 'dat': dat, 'train_dat': train_dat, 'log_dep': log_dep}


def calc_perror(actual, prediction):
    ape = 100 * np.abs(actual - prediction) / actual
    aape = np.round(np.nanmean(ape), 2)
    mape = np.round(np.nanmedian(ape), 2)
    return {'aape': aape, 'mape': mape}

def feature_predict(model, dat, train_dat, log_dep):
    key_cols = ['site', 'value']
    # Fit main model and predictions
    main_model = lgb.LGBMRegressor(**model.best_params_).\
        fit(y = train_dat['value'], X = train_dat.drop(columns = key_cols))
    predict_array = main_model.predict(dat.drop(columns=key_cols))
    pred_dat = pd.concat([
        dat[key_cols],
        pd.DataFrame(predict_array, columns=['ws'])],
        axis=1)
    # Calc OOS predictions + error
    train_a, train_b = train_test_split(train_dat, test_size=0.5)
    train_a, train_b = [train_a.reset_index(), train_b.reset_index()]
    split_model = lgb.LGBMRegressor(**model.best_params_). \
        fit(y=train_a['value'], X=train_a.drop(columns=key_cols))
    predict_array_b = split_model.predict(train_b.drop(columns=key_cols))
    split_model = lgb.LGBMRegressor(**model.best_params_). \
        fit(y=train_b['value'], X=train_b.drop(columns=key_cols))
    predict_array_a = split_model.predict(train_a.drop(columns=key_cols))

    train_a = pd.concat([train_a[['site']], pd.DataFrame(predict_array_a, columns=['oos'])], axis=1)
    train_b = pd.concat([train_b[['site']], pd.DataFrame(predict_array_b, columns=['oos'])], axis=1)

    pred_dat = pred_dat.merge(
        pd.concat([train_a, train_b], axis=0),
        on='site',
        how='outer')
    if(log_dep):
        pred_dat[['ws', 'oos']] = pred_dat[['ws', 'oos']].apply(lambda x: np.expm1(x))
    error = calc_perror(pred_dat['value'], pred_dat['oos'])

    return {'model': main_model, 'predictions': pred_dat, 'error': error}

def tab_importance(model, **kwargs):
    imp_dat = pd.DataFrame({'feature': model.feature_name_, 'importance': model.feature_importances_})
    imp_dat['relative'] = np.round(100 * imp_dat['importance'] / np.sum(imp_dat['importance']))
    imp_dat = imp_dat.sort_values(by='relative', ascending=False).reset_index(drop=True)
    return imp_dat
