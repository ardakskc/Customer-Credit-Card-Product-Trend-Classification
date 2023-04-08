import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

KATEGORIK_SARTI = 10

###############################################################
"""classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]"""
#################################################################

def get_categorical_and_numeric_columns(df, list_of_variables=None):
    x = df.dtypes
    x = pd.DataFrame(x, columns=['types'])
    if (list_of_variables != None):
        x = x[x.index.isin(list_of_variables)]

    binary_kategorik = []
    int_col = x[(x['types'] == 'int64')].index
    int_col = int_col.dropna()
    int_col = int_col.tolist()

    for i in int_col:
        n = len(pd.unique(df[i]))
        # nonUnique > 10 ise numerik olarak al. isimden de filtrele.
        if n < KATEGORIK_SARTI and i.find("SAYI") == -1 and i.find("SY") == -1 and i.find("TUT") == -1:
            binary_kategorik.append(i)

    for i in binary_kategorik:
        int_col.remove(i)

    num_cols = x[(x['types'] == 'float64')].index
    num_cols = num_cols.dropna()
    num_cols = num_cols.tolist()
    for i in int_col:
        num_cols.append(i)

    cat_cols = x[(x['types'] == 'object')].index
    cat_cols = cat_cols.dropna()
    cat_cols = cat_cols.tolist()
    for i in binary_kategorik:
        cat_cols.append(i)

    date_cols = x[(x['types'] == 'datetime64[ns]')].index
    date_cols = date_cols.dropna()
    date_cols = date_cols.tolist()

    print('Kategorik ve nümerik veriler döndürüldü.')

    return cat_cols, num_cols, date_cols




def model_metrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    ##Visualize conf matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()

    TP = cm[0][0]
    FP = cm[0][1]
    TN = cm[1][1]
    FN = cm[1][0]

    arr = precision_recall_fscore_support(y_test, y_pred, average='macro')
    # print("Precision: {:.2f}\nRecall: {:.2f}\nFScore: {:.2f}".format(arr[0], arr[1], arr[2]))
    print(classification_report(y_test, y_pred))
    return arr[0], arr[1], TP, FP, TN, FN

#pickle and pmml arastir.


df = pd.read_csv(r'SON_modified_kk.csv', sep=';', encoding='windows-1254', low_memory=False)
df.set_index('REF_TCKNVKN_ID', inplace=True)

######SCALING ISLEMLERI#######
scaler = MinMaxScaler()
cat, num, date = get_categorical_and_numeric_columns(df)
# Min-max scaling on numeric columns
df[num] = scaler.fit_transform(df[num])
##############################

X = df.copy()
y = X['KK_ACIK_FLAG'].copy()
X.drop('KK_ACIK_FLAG', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.33)

b_tr = y_train.value_counts(normalize=True) * 100  ####Train test balans oranları aynı.
b_tst = y_test.value_counts(normalize=True) * 100

model_sonuc = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'TP', 'FP', 'TN', 'FN'])


############################################################
#####################MODELLEME##############################
############################################################




#########################
###Logistic Regression###
print("###Logistic Regression###")
log_reg = LogisticRegression(random_state=0)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(X_test, y_test)))
acc = log_reg.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'LogisticRegression', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)
print("#########################")
#########################################################################




###SUPPORT VECTOR MACHINE###
print("###SUPPORT VECTOR MACHINE###")
from sklearn.svm import SVC

svc = SVC(kernel='poly')
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

print('Accuracy of Support Vector classifier on test set: {:.2f}'.format(svc.score(X_test, y_test)))
acc = svc.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'SVM', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("###########################")
##############################################################



###KNN###
from sklearn.neighbors import KNeighborsClassifier

print("###KNN###")
###########KNN HİPERPARAMETRE OPTİMİZASYONU - GRIDSEARCHCV##############
"""n_neighbors = list(range(5, 15))
p = [1, 2]
metric = ['minkowski', 'euclidean']

hyperparameters = dict(n_neighbors=n_neighbors, p=p, metric=metric)
knn = KNeighborsClassifier()

knn_2 = GridSearchCV(knn, hyperparameters, cv=10)

knn_2.fit(X_train, y_train)

print('Best p: ', knn_2.best_estimator_.get_params()['p'])
print('Best n_neighbors: ', knn_2.best_estimator_.get_params()['n_neighbors'])
print('Best distance metric: ', knn_2.best_estimator_.get_params()['metric'])"""
#########################################################################
###Optimizasyon sonuclari
"""Best p:  1
Best n_neighbors:  9
Best distance metric:  minkowski"""

knn = KNeighborsClassifier(leaf_size=30,
                           p=1,
                           n_neighbors=9,
                           metric='minkowski')

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Accuracy of K Neighbors classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))

acc = knn.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'KNN', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("###########")
###############################################################




####DECISIONTREECLASSIFIER#####
print("####DECISION TREE#####")
from sklearn.tree import DecisionTreeClassifier

######HIPERPARAMETRE OPTIMIZASYONU#################
"""dtc = DecisionTreeClassifier()

criterion = ['gini', 'entropy', 'log_loss']
hyperparameters = dict(criterion=criterion)
dtc_opt = GridSearchCV(dtc, hyperparameters, cv=10)
dtc_opt.fit(X_train, y_train)

print('Best criterion: {}'.format(dtc_opt.best_estimator_.get_params()['criterion']))"""
#####################################################
#####Best criterion: entropy

dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(dtc.score(X_test, y_test)))

acc = dtc.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'Decision Tree', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("###########")
############################################################




########RANDOM FOREST ###########
print("########RANDOM FOREST ###########")

from sklearn.ensemble import RandomForestClassifier

###########Hiperparametre Optimizasyonu##############
"""rfc = RandomForestClassifier()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=500, num=2)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = dict(n_estimators=n_estimators,
                   max_features=max_features,
                   bootstrap=bootstrap)

rfc_opt = GridSearchCV(rfc, random_grid, cv=10)
rfc_opt.fit(X_train, y_train)

print('Best n_estimators: {}'.format(rfc_opt.best_estimator_.get_params()['n_estimators']))
print('Best max_features: ', rfc_opt.best_estimator_.get_params()['max_features'])
print('Best bootstrap: ', rfc_opt.best_estimator_.get_params()['bootstrap'])"""
########################################################
####Sonuclar####
"""Best n_estimators: 500
Best max_features:  sqrt
Best bootstrap:  False"""

rfc = RandomForestClassifier(n_estimators=500,
                             max_features='sqrt',
                             bootstrap=False)

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(rfc.score(X_test, y_test)))

acc = rfc.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'Random Forest', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("##################################")
####################################################################





####ADABOOST####
print("####ADABOOST####")
from sklearn.ensemble import AdaBoostClassifier

##Decision tree parametreleri vir onceki decision tree modelinden alındı.
ada_clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, criterion='gini'),
                             n_estimators=70,
                             learning_rate=0.5,
                             algorithm='SAMME.R',
                             random_state=1)

ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)

print('Accuracy of AdaBoost classifier on test set: {:.2f}'.format(ada_clf.score(X_test, y_test)))

acc = ada_clf.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'AdaBoost', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("##################################")
######################################################################




####XGBOOST#####
print("####XGBOOST####")
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

"""space = {'max_depth': hp.quniform("max_depth", 3, 18, 1),
         'gamma': hp.uniform('gamma', 1, 9),
         'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
         'reg_lambda': hp.uniform('reg_lambda', 0, 1),
         'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
         'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
         'n_estimators': 180,
         'seed': 0
         }


def objective(space):
    clf = xgb.XGBClassifier(
        n_estimators=space['n_estimators'], max_depth=int(space['max_depth']), gamma=space['gamma'],
        reg_alpha=int(space['reg_alpha']), min_child_weight=int(space['min_child_weight']),
        colsample_bytree=int(space['colsample_bytree']))

    evaluation = [(X_train, y_train), (X_test, y_test)]

    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10, verbose=False)

    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
    print("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK}


trials = Trials()

best_hyperparam = fmin(fn=objective,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=100,
                       trials=trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparam)"""

##BEST HYPERPARAMETERS OUTPUT:
"""The best hyperparameters are :
{'colsample_bytree': 0.5654522141387407, 'gamma': 6.10208677101715, 
 'max_depth': 4.0, 'min_child_weight': 4.0, 'reg_alpha': 41.0, 'reg_lambda': 0.19354383660163538}"""

xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        booster='gbtree', nthread=3, colsample_bytree=0.6,
                        gamma=6, max_depth=4, min_child_weight=4, reg_alpha=41, reg_lambda=0.2)

xgb_clf.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

print('Accuracy of XGBOOST classifier on test set: {:.2f}'.format(xgb_clf.score(X_test, y_test)))

acc = xgb_clf.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'XGBoost', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("##################################")
#########################################################################




####CatBoost####
from catboost import CatBoostClassifier

print("####CatBoost####")

###PARAMETRE OPTIMIZASYONU
"""CBC = CatBoostClassifier()
parameters = dict(depth=[6, 8],
                  learning_rate=[0.02, 0.04],
                  l2_leaf_reg=[1, 3, 5, ],
                  iterations=[100, 150, 200],
                  loss_function=['LogLoss', 'CrossEntropy']
                  )

Grid_CBC = GridSearchCV(CBC,parameters, cv = 2, n_jobs=-1)

Grid_CBC.fit(X_train, y_train)
print("\n The best score across ALL searched params:\n", Grid_CBC.best_score_)
print("\n The best parameters across ALL searched params:\n", Grid_CBC.best_params_)"""

####OPTIMIZE PARAMETRELER######
""" The best score across ALL searched params:
 0.9284612418048592
 The best parameters across ALL searched params:
 {'depth': 6, 'iterations': 200, 'l2_leaf_reg': 1, 'learning_rate': 0.02, 'loss_function': 'CrossEntropy'}
"""

CBC = CatBoostClassifier(depth=6, iterations=150, verbose=False,
                         l2_leaf_reg=2, learning_rate=0.02, loss_function='CrossEntropy')
CBC.fit(X_train, y_train)
y_pred = xgb_clf.predict(X_test)

print('Accuracy of CatBoost classifier on test set: {:.2f}'.format(CBC.score(X_test, y_test)))

acc = CBC.score(X_test, y_test)
pre, rec, tp, fp, tn, fn = model_metrics(y_test, y_pred)

temp = {'Model': 'CatBoost', 'Accuracy': acc, 'Precision': pre,
        'Recall': rec, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
model_sonuc = model_sonuc.append(temp, ignore_index=True)

print("##################################")

model_sonuc.to_csv('model_sonuc.csv', index=False, sep=';',encoding='windows-1254')
###################################################################################

