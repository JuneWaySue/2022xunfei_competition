import os
import math
import random

import pandas as pd
import numpy as np
from matplotlib.pyplot import plot, show

import lightgbm as lgb
import xgboost as xgb
import catboost as cat

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def calc_acc(y_true,y_pred):
    acc=0
    howmanay=len(y_true)//209
    for i in range(howmanay):
        start=i*209
        end=(i+1)*209
        target=y_true[start:end].tolist()
        predict=y_pred[start:end].tolist()
        for j in range(len(target)):
            t=target[j]
            p=predict[j]
            if t == 0:
                continue
            acc+=(1-np.abs(p-t)/t)*(t/np.sum(target))
    return acc/howmanay

def get_data():
    seed_everything(2022)
    fusai = True
    root_path='../input/2022xunfei-product-sales-data/'
    product_month_order_train_path = root_path+f'product_month_order_train{"_fusai" if fusai else ""}.csv'
    product_need_train_path = root_path+f'product_need_train{"_fusai" if fusai else ""}.csv'
    product_month_order_test_path = root_path+f'product_month_order_test{"_fusai" if fusai else ""}.csv'
    product_need_test_path = root_path+f'product_need_test{"_fusai" if fusai else ""}.csv'

    need_df=pd.read_csv(product_need_train_path)
    order_df=pd.read_csv(product_month_order_train_path).rename(columns={'year':'year_id','month':'month_id'})
    need_df['date']=pd.to_datetime(need_df['date'])
    need_df=pd.DataFrame(
        [{'product_id':product_id,'date':date} for product_id in need_df.product_id.unique()
        for date in pd.date_range(need_df.date.min(),need_df.date.max())]).merge(need_df,on=['product_id','date'],how='left')
    need_df=need_df.fillna(0)
    need_df['year_id']=need_df['date'].dt.year
    need_df['month_id']=need_df['date'].dt.month
    need_df=need_df.groupby(['product_id','year_id','month_id'],as_index=False).agg({'is_sale_day':['nunique','max','sum'],'label':[('','sum')]})
    need_df.columns=['_'.join(i) if i[1] != '' else i[0] for i in need_df.columns.to_list()]
    train_df=need_df.merge(order_df,on=['product_id','year_id','month_id'],how='left')

    need_df=pd.read_csv(product_need_test_path)
    order_df=pd.read_csv(product_month_order_test_path).rename(columns={'year':'year_id','month':'month_id'})
    need_df['date']=pd.to_datetime(need_df['date'])
    need_df=pd.DataFrame(
        [{'product_id':product_id,'date':date} for product_id in need_df.product_id.unique()
        for date in pd.date_range(need_df.date.min(),need_df.date.max())]).merge(need_df,on=['product_id','date'],how='left')
    need_df=need_df.fillna(0)
    need_df['year_id']=need_df['date'].dt.year
    need_df['month_id']=need_df['date'].dt.month
    need_df=need_df.groupby(['product_id','year_id','month_id'],as_index=False).agg({'is_sale_day':['nunique','max','sum']})
    need_df.columns=['_'.join(i) if i[1] != '' else i[0] for i in need_df.columns.to_list()]
    test_df=need_df.merge(order_df,on=['product_id','year_id','month_id'],how='left')
    
    data=pd.concat([train_df,test_df],ignore_index=True)
    data=data.sort_values(by=['year_id','month_id', 'product_id']).reset_index(drop=True)
    data['time_id'] = list(map(lambda x,y:(x-2018)*12+y-1,data['year_id'],data['month_id']))
    data['jidu_id']=((data['month_id']-1)/3+1).map(int)

    data['is_sale_day_max']=data['is_sale_day_max'].astype(int)
    data['type']=LabelEncoder().fit_transform(data['type'])
    data['label'] = np.log1p(data['label'])
    # data['label'] = list(map(lambda x : x if x==np.NAN else math.log(x+1,2),data['label']))

    feats_cols=data.columns.to_list()
    
    for i in range(1, 17):
        for f in ['label', 'order', 'start_stock', 'end_stock']:
            data[f+'_shift_%d'%i] = data.groupby('product_id')[f].shift(i+2)
            if i <= 12:
                feats_cols.append(f+'_shift_%d'%i)

    for i in [3, 6, 12]:
        for f in ['label', 'order', 'start_stock', 'end_stock']:
            data[f+'_mean_%d'%i] = data[[f+'_shift_%d'%i for i in range(1, i+1)]].mean(axis=1)
            data[f+'_std_%d'%i] = data[[f+'_shift_%d'%i for i in range(1, i+1)]].std(axis=1)
            data[f+'_median_%d'%i] = data[[f+'_shift_%d'%i for i in range(1, i+1)]].median(axis=1)
            feats_cols.extend([f+'_mean_%d'%i,f+'_std_%d'%i,f+'_median_%d'%i])

    category_cols=[
        'product_id','year_id','month_id','time_id','jidu_id','is_sale_day_nunique','is_sale_day_max','type'
    ]

    print('data.shape：',data.shape)
    return data,feats_cols,category_cols

def train_model_with_nfold(data,train,test,feat_cols,feats_cols,category_cols,
    lgb_params=None,xgb_params=None,cat_params=None,model_types=['lgb','xgb','cat'],fold_num=5,
    seeds=[2022],stratified=True,num_boost_round=10000,early_stopping_rounds=200,verbose=200,
    un_select_cols=[]
    ):

    score_lgb = np.zeros(fold_num)
    score_xgb = np.zeros(fold_num)
    score_cat = np.zeros(fold_num)
    score = np.zeros(fold_num)
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    oof = np.zeros(len(train))
    pred_y = pd.DataFrame()

    for seed in seeds:
        for model_type in model_types:
            if stratified:
                kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
            else:
                kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
            if model_type == 'cat':
                feat_cols = [col for col in feats_cols if col not in un_select_cols+['label']]
                np.random.shuffle(feat_cols)
            if model_type == 'xgb':
                if 1001 in train['product_id'].values:
                    LE=LabelEncoder()
                    for col in category_cols:
                        LE.fit(data[col].astype(str))
                        train[col]=LE.transform(train[col].astype(str))
                        test[col]=LE.transform(test[col].astype(str))
            else:
                train = data[data['label'].notna()].reset_index(drop=True)
                test = data[data['label'].isna()].reset_index(drop=True)
            for fold, (train_idx, val_idx) in enumerate(kf.split(train[feat_cols], train['product_id'])):
                print(f'-----------------fold：{fold+1} -----------------seed：{seed} -----------------model_type：{model_type}')
                if model_type == 'lgb':
                    lgb_params['seed']=seed
                    tra = lgb.Dataset(train.loc[train_idx, feat_cols],train.loc[train_idx, 'label'])
                    val = lgb.Dataset(train.loc[val_idx, feat_cols],train.loc[val_idx, 'label'])
                    model = lgb.train(lgb_params, tra, valid_sets=[val], num_boost_round=num_boost_round,categorical_feature=category_cols,
                                    callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(verbose)])

                    score_lgb[fold]=model.best_score['valid_0']['rmse'] / len(seeds)
                    score[fold]=model.best_score['valid_0']['rmse'] / len(seeds) / len(model_types)
                    oof_lgb[val_idx] += model.predict(train.loc[val_idx, feat_cols], num_iteration=model.best_iteration) / len(seeds)
                    oof[val_idx] += model.predict(train.loc[val_idx, feat_cols], num_iteration=model.best_iteration) / len(seeds) / len(model_types)
                    pred_y[f'fold{fold}_seed{seed}_{model_type}'] = model.predict(test[feat_cols], num_iteration=model.best_iteration)
                elif model_type == 'xgb':
                    xgb_params['seed']=seed
                    train_matrix = xgb.DMatrix(train.loc[train_idx, feat_cols] , label=train.loc[train_idx, 'label'])
                    valid_matrix = xgb.DMatrix(train.loc[val_idx, feat_cols] , label=train.loc[val_idx, 'label'])
                    watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
                    model = xgb.train(xgb_params, train_matrix, num_boost_round=num_boost_round, evals=watchlist, verbose_eval=verbose, early_stopping_rounds=early_stopping_rounds)

                    score_xgb[fold]=model.best_score / len(seeds)
                    score[fold]=model.best_score / len(seeds) / len(model_types)
                    oof_xgb[val_idx] += model.predict(valid_matrix,iteration_range=(0,model.best_iteration+1)) / len(seeds)
                    oof[val_idx] += model.predict(valid_matrix,iteration_range=(0,model.best_iteration+1)) / len(seeds) / len(model_types)
                    pred_y[f'fold{fold}_seed{seed}_{model_type}'] = model.predict(xgb.DMatrix(test[feat_cols]),iteration_range=(0,model.best_iteration+1))
                else:
                    cat_params['random_seed']=seed
                    model = cat.CatBoostRegressor(num_boost_round=num_boost_round, **cat_params)
                    trn_x = train.loc[train_idx, feat_cols]
                    trn_y = train.loc[train_idx, 'label']
                    val_x = train.loc[val_idx, feat_cols]
                    val_y = train.loc[val_idx, 'label']
                    model.fit(trn_x, trn_y, eval_set=(val_x, val_y),cat_features=category_cols, use_best_model=True, verbose=verbose)
                    
                    score_cat[fold] = model.best_score_['validation']['RMSE'] / len(seeds)
                    score[fold] = model.best_score_['validation']['RMSE'] / len(seeds) / len(model_types)
                    oof_cat[val_idx] += model.predict(val_x) / len(seeds)
                    oof[val_idx] += model.predict(val_x) / len(seeds) / len(model_types)
                    pred_y[f'fold{fold}_seed{seed}_{model_type}'] = model.predict(test[feat_cols])

    print(f'score_lgb={score_lgb.mean()}\nscore_xgb={score_xgb.mean()}\nscore_cat={score_cat.mean()}\nscore={score.mean()}')

    return oof_lgb,oof_xgb,oof_cat,oof,pred_y

def RidgeCV(train,oof_lgb,oof_xgb,oof_cat,pred_y,fold_num=5,seed=2022):
    oof=pd.DataFrame([oof_lgb,oof_xgb,oof_cat]).T
    oof.columns=['lgb','xgb','cat']
    oof['product_id']=train['product_id']
    oof['label']=train['label']

    y=pd.DataFrame([pred_y[[i for i in pred_y.columns if 'lgb' in i]].mean(1),pred_y[[i for i in pred_y.columns if 'xgb' in i]].mean(1),pred_y[[i for i in pred_y.columns if 'cat' in i]].mean(1)]).T
    y.columns=['lgb','xgb','cat']

    kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    oof_res=np.zeros(len(oof))
    prediction=np.zeros(len(y))
    for fold, (train_idx, val_idx) in enumerate(kf.split(oof[['lgb','xgb','cat']], oof['product_id'])):
        train_x=oof.loc[train_idx,['lgb','xgb','cat']]
        train_y=oof.loc[train_idx,'label']
        
        valid_x=oof.loc[val_idx,['lgb','xgb','cat']]
        valid_y=oof.loc[val_idx,'label']
        
        model = Ridge(random_state=2022)
        model.fit(train_x,train_y)
        
        oof_res[val_idx]=model.predict(valid_x)
        prediction+=model.predict(y)/fold_num

    return oof_res,prediction

def main():
    data,feats_cols,category_cols=get_data()
    train = data[data['label'].notna()].reset_index(drop=True)
    test = data[data['label'].isna()].reset_index(drop=True)

    un_select_cols=[]
    feat_cols = [col for col in feats_cols if col not in un_select_cols+['label']]
    np.random.shuffle(feat_cols)
    print('feat_cols.len：',len(feat_cols))

    #lgb
    lgb_params={
        'lambda_l2': 3.1841775244883856,
        'learning_rate': 0.011388441559132073,
        'max_depth': 28,
        'min_child_weight': 3.404010799108265,
        'num_leaves': 77,
        'feature_pre_filter': False,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'seed': 2022,
        'n_jobs': -1,
        'verbose': -1
    }

    # xgb
    xgb_params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.07660224004250459,
        'max_depth': 9,
        'min_child_weight': 11.269262917397498
    }

    # cat
    cat_params = {
        'learning_rate': 0.3,
        'loss_function':'RMSE',
        'depth': 6,
        'l2_leaf_reg': 3,
        'od_type': 'Iter',
        'od_wait': 200,
        'allow_writing_files': False
    }

    oof_lgb,oof_xgb,oof_cat,oof,pred_y=train_model_with_nfold(data,train,test,feat_cols,feats_cols,category_cols,
        lgb_params=lgb_params,xgb_params=xgb_params,cat_params=cat_params,model_types=['lgb','xgb','cat'],fold_num=5,
        seeds=[2022],stratified=True,num_boost_round=10000,early_stopping_rounds=200,verbose=200,
        un_select_cols=[]
    )

    oof_res,prediction=RidgeCV(train,oof_lgb,oof_xgb,oof_cat,pred_y,fold_num=5,seed=2022)

    train['target_weight'] = train['label'] / train.groupby(['year_id', 'month_id'])['label'].transform('sum')
    train['oof'] = np.expm1(oof_res)
    train['oof'] = train['oof'].map(lambda x: x if x >= 0 else 0)
    train['oof'] = train['oof'].round()
    train['label'] = np.expm1(train['label'])
    print(np.expm1(prediction).sum())
    print(math.sqrt(mean_squared_error(train['label'],train['oof'])))
    # train['oof'] = list(map(lambda x : x if x==np.NAN else (2**(x))-1, oof))
    # train['label'] = list(map(lambda x : x if x==np.NAN else (2**(x))-1, train['label']))

    score1 = np.sum((1 - np.abs(train['label']-train['oof']) / (train['label'])) * train['target_weight']) / (len(train)//209)
    print(score1)
    print(np.mean(np.abs(train['label']-train['oof'])/(train['label']+1)))
    print(calc_acc(train['label'],train['oof']))
    plot(train['label'])
    plot(train['oof'])
    show()

    test['label'] = np.expm1(prediction)
    # test['label'] = list(map(lambda x : x if x==np.NAN else (2**(x))-1, pred_y.mean(axis=1).values))

    sub=data[data['label'].isna()].reset_index(drop=True)[['product_id','year_id','month_id','label']].copy()
    sub['label'] = test['label'].map(lambda x: x if x >= 0 else 0)

    sub['month']=sub['year_id'].astype(str)+'-'+sub['month_id'].apply(lambda x:f'{x:02}')
    sub=sub[['month','product_id','label']]
    sub['label']=sub['label'].round()
    
    return sub


if __name__ == '__main__':
    sub = main()
    sub.to_csv('submit.csv',index=False)