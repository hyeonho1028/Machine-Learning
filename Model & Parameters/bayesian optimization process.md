#### Kaggle

https://www.kaggle.com/fayzur/lgb-bayesian-parameters-finding-rank-average



```
print(LGB_BO.max['target'])
print(LGB_BO.max['params'])

LGB_BO.probe(
    params={'bagging_fraction': 0.41574271615734787, 
            'bagging_freq': 9.003414752912288, 
            'feature_fraction': 0.4607111724056043, 
            'lambda_l1': 0.34089270630775115, 
            'lambda_l2': 0.05391456744494457, 
            'learning_rate': 0.014437740917852413, 
            'max_depth': 14.150526604709201, 
            'min_data_in_leaf': 7.631758698632379, 
            'num_leaves': 5.015082332845268},
    lazy=True, # 
)

LGB_BO.maximize(init_points=0, n_iter=0) # remember no init_points or n_iter

for i, res in enumerate(LGB_BO.res):
    print("Iteration {}: \n\t{}".format(i, res))


LGB_BO.max['target']
LGB_BO.max['params']


# {'bagging_fraction': 0.41574271615734787,
#  'bagging_freq': 9.003414752912288,
#  'feature_fraction': 0.4607111724056043,
#  'lambda_l1': 0.34089270630775115,
#  'lambda_l2': 0.05391456744494457,
#  'learning_rate': 0.014437740917852413,
#  'max_depth': 14.150526604709201,
#  'min_data_in_leaf': 7.631758698632379,
#  'num_leaves': 5.015082332845268}
random_state=1028
param = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : int(LGB_BO.max['params']['max_depth']),
    "num_leaves" : int(LGB_BO.max['params']['num_leaves']),
    "min_data_in_leaf": int(LGB_BO.max['params']['num_leaves']),
    "bagging_freq": int(LGB_BO.max['params']['bagging_freq']),
#         "min_sum_heassian_in_leaf": min_sum_heassian_in_leaf,
    "learning_rate" : LGB_BO.max['params']['learning_rate'],
    "bagging_fraction" : LGB_BO.max['params']['bagging_fraction'],
    "feature_fraction" : LGB_BO.max['params']['feature_fraction'],
    "tree_learner": "serial",
    "boost_from_average": "false",
    "lambda_l1" : LGB_BO.max['params']['lambda_l1'],
    "lambda_l2" : LGB_BO.max['params']['lambda_l2'],
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}    

splits = 5
skf = StratifiedKFold(n_splits=splits, random_state=1028)

oof = np.zeros(len(df4))
predictions = np.zeros(len(df_test))

for trn_idx, val_idx in skf.split(df4.drop(columns='target'), df4['target']):
    
    X_train, y_train = df4.drop(columns='target').loc[trn_idx], df4['target'][trn_idx]
    X_test, y_test = df4.drop(columns='target').loc[val_idx], df4['target'][val_idx]
    
    xg_train = lgb.Dataset(X_train, label=y_train)
    xg_valid = lgb.Dataset(X_test, label=y_test)

    clf = lgb.train(param, xg_train, 100000, valid_sets = [xg_train, xg_valid], early_stopping_rounds = 3000, verbose_eval=1000)

    oof[val_idx] = clf.predict(X_test, num_iteration=clf.best_iteration)
    predictions += clf.predict(df_test, num_iteration=clf.best_iteration)/splits

print(roc_auc_score(df4['target'], oof))
```





정처기

링크

<https://quizlet.com/388261501/%EC%99%B8%EC%9A%B0%EC%9E%90-flash-cards/?i=2105w9&x=1jqY>