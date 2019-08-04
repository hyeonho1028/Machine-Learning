##### Lgbm document



https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/sklearn.html

parameter

lgbm같은 경우 num_leaves라는 하이퍼파라미터가 있어 max_depth에 굳이 제한을 줄 필요가 없어 -1이 기본값으로 설정되어있습니다! 극한의 튜닝을 위해서는 둘다 동시에 설정하셔도 되는데 num_leaves만 이용하시는걸 추천드려요ㅎㅎ (lgbm은 depth를 기준으로 하는 level-wise 학습이아니고 leaf-wise 학습이기 때문입니다!)

##### XGBOOST

Parameter document : https://xgboost.readthedocs.io/en/latest/parameter.html