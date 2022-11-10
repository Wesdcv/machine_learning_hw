! wget https://raw.githubusercontent.com/girafe-ai/ml-mipt/21f_made/homeworks/Lab1_ML_pipeline_and_SVM/car_data.csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from tqdm import tqdm
import seaborn as sns
dataset = pd.read_csv('car_data.csv', delimiter=',', header=None).values
data = dataset[:, :-1].astype(int)
target = dataset[:, -1]

print(data.shape, target.shape)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.35)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
X_train_pd = pd.DataFrame(X_train)

# First 15 rows of our dataset.
X_train_pd.head(5)
X_train_pd.describe()
X_train_pd.info()
normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm =  normalizer.transform(X_test)
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
np.logspace(-3, 3, 10)
%%time
# подбираем гиперпараметры
log_reg_model = LogisticRegression(multi_class='multinomial', solver='saga', tol=1e-3, max_iter=500)

grid = {'C': np.logspace(-3, 3, 10),
        'penalty': ['l1', 'l2']}

grid_log_reg = GridSearchCV(log_reg_model, grid, scoring=['accuracy', 'f1_weighted'], refit='accuracy', cv=5)
grid_log_reg.fit(X_train_norm, y_train_encoded)
logreg_cv_results = pd.DataFrame(grid_log_reg.cv_results_)
logreg_cv_results[['mean_test_accuracy', 'mean_test_f1_weighted']].sort_values(by=['mean_test_accuracy', 'mean_test_f1_weighted'], ascending=False)[:5]
grid_log_reg.best_params_
grid_log_reg.best_score_
# accuracy and f1 on train
best_model = grid_log_reg.best_estimator_

y_pred = best_model.predict(X_train_norm)
accuracy = accuracy_score(y_train_encoded, y_pred)
f1 = f1_score(y_train_encoded, y_pred, average='weighted')
print("TRAIN ACCURACY: {}\nTRAIN F1_SCORE: {}".format(accuracy, f1))
# accuracy and f1 on test
y_pred = best_model.predict(X_test_norm)
accuracy = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print("TEST ACCURACY: {}\nTEST F1_SCORE: {}".format(accuracy, f1))
# You might use this command to install scikit-plot. 
# Warning, if you a running locally, don't call pip from within jupyter, call it from terminal in the corresponding 
# virtual environment instead

! pip install scikit-plot
from scikitplot.metrics import plot_roc

plot_roc(y_test_encoded, best_model.predict_proba(X_test_norm))
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train_norm)
fig = plt.figure(figsize=(15, 10))

feature_num = X_train_norm.shape[1]

plt.plot(np.arange(1, feature_num + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, feature_num + 1))
for x, y in zip(np.arange(1, feature_num + 1), pca.explained_variance_ratio_.cumsum()):
  plt.annotate(f'{y:.2}', (x, y), textcoords="offset points",
               xytext=(0,10),
               ha='center')
plt.title('Explained variance plot')
plt.xlabel('Number of components')
plt.ylabel('Explained variance ratio pct')
plt.xlim(1, feature_num)
### YOUR CODE HERE
np.cumsum(pca.explained_variance_ratio_)
pca = PCA(n_components=12)
decomposed_train = pca.fit_transform(X_train_norm)
decomposed_test = pca.transform(X_test_norm)
from sklearn.pipeline import Pipeline
%%time
pipe = Pipeline([('scaler', StandardScaler()), 
                 ('dim_reduction', PCA(n_components=12)), 
                 ('clf', LogisticRegression(multi_class='multinomial', solver='saga', tol=1e-3))
                 ])

N_FEATURES_OPTIONS = [4, 8, 12]
C_OPTIONS = np.logspace(-3, 3, 10)
PENALTY_OPTIONS = ['l1', 'l2']

param_grid = [
    {
        "dim_reduction__n_components": N_FEATURES_OPTIONS,
        "clf__C": C_OPTIONS,
        "clf__penalty": PENALTY_OPTIONS
    },
]

scoring = ['accuracy', 'f1_weighted']

grid = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=5, return_train_score=True)
grid.fit(X_train, y_train_encoded)
grid.best_params_
#TRAIN
y_pred = grid.predict(X_train)
accuracy = accuracy_score(y_train_encoded, y_pred)
f1 = f1_score(y_train_encoded, y_pred, average='weighted')
print("ACCURACY: {}\nF1_SCORE: {}".format(accuracy, f1))
# TEST
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print("TEST ACCURACY: {}\nTEST F1_SCORE: {}".format(accuracy, f1))
plot_roc(y_test_encoded, grid.predict_proba(X_test))
from sklearn.tree import DecisionTreeClassifier
%%time
pipe = Pipeline([('scaler', StandardScaler()), 
                 ('dim_reduction', PCA(n_components=12)), 
                 ('clf', DecisionTreeClassifier())])

MAMX_DEPTH_OPTIONS = np.linspace(1, 50, 50)


param_grid = [
    {
        "clf__max_depth": MAMX_DEPTH_OPTIONS
    }
]

scoring = ['accuracy', 'f1_weighted']

grid = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, refit='accuracy', cv=5, return_train_score=True)
grid.fit(X_train, y_train_encoded)
grid.best_params_
#TRAIN
y_pred = grid.predict(X_train)
accuracy = accuracy_score(y_train_encoded, y_pred)
f1 = f1_score(y_train_encoded, y_pred, average='weighted')
print("ACCURACY: {}\nF1_SCORE: {}".format(accuracy, f1))
# TEST
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print("TEST ACCURACY: {}\nTEST F1_SCORE: {}".format(accuracy, f1))
plot_roc(y_test_encoded, grid.predict_proba(X_test))
# YOUR CODE HERE
from sklearn.ensemble import BaggingClassifier
from tqdm import tqdm

n_min = 2
n_max = 100
step = 5

alg_nums = np.arange(n_min, n_max, step)
acc_scores_train = np.array([])
f1_scores_train = np.array([])
acc_scores_test = np.array([])
f1_scores_test = np.array([])

for n_estimators in tqdm(np.arange(n_min, n_max, step)):
  pipe = Pipeline([('scaler', StandardScaler()), 
                  ('dim_reduction', PCA(n_components=12)), 
                  ('clf', BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=n_estimators))
                  ])

  pipe.fit(X_train, y_train_encoded)

  y_pred_test = pipe.predict(X_test)
  y_pred_train = pipe.predict(X_train)

  acc_scores_train = np.append(acc_scores_train, accuracy_score(y_train_encoded, y_pred_train))
  f1_scores_train = np.append(f1_scores_train, f1_score(y_train_encoded, y_pred_train, average='weighted'))
  acc_scores_test = np.append(acc_scores_test, accuracy_score(y_test_encoded, y_pred_test))
  f1_scores_test = np.append(f1_scores_test, f1_score(y_test_encoded, y_pred_test, average='weighted'))
fig=plt.figure()
fig.set_size_inches((15, 10))
fig.show()
ax = fig.add_subplot(111)
ax.set_xticks(alg_nums)
ax.set_xlabel('Estimator num', fontdict={'fontsize': 15})
ax.set_ylabel('Score', fontdict={'fontsize': 15})

params = {'legend.fontsize': 14,
          'figure.figsize': (15, 10),
        }
plt.rcParams.update(params)


ax.plot(alg_nums, acc_scores_train, label='TRAIN ACCURACY')
ax.plot(alg_nums, f1_scores_train, label='TRAIN F1')
ax.plot(alg_nums, acc_scores_test, label='TEST ACCURACY')
ax.plot(alg_nums, f1_scores_test, label='TEST F1')


plt.title('Score plots w.r.t. tree ensemble size', fontdict={'fontsize': 20})
plt.legend(loc='lower right');
alg_nums = np.arange(n_min, n_max, step)
acc_scores_train = np.array([])
f1_scores_train = np.array([])
acc_scores_test = np.array([])
f1_scores_test = np.array([])

for n_estimators in tqdm(np.arange(n_min, n_max, step)):
  pipe = Pipeline([('scaler', StandardScaler()), 
                  ('dim_reduction', PCA(n_components=12)), 
                  ('clf', BaggingClassifier(base_estimator=LogisticRegression(
                      multi_class='multinomial', 
                      solver='saga', 
                      tol=1e-3, C=0.46415888336127775, 
                      penalty='l1'), n_estimators=n_estimators))])
  
  pipe.fit(X_train, y_train_encoded)

  y_pred_test = pipe.predict(X_test)
  y_pred_train = pipe.predict(X_train)

  acc_scores_train = np.append(acc_scores_train, accuracy_score(y_train_encoded, y_pred_train))
  f1_scores_train = np.append(f1_scores_train, f1_score(y_train_encoded, y_pred_train, average='weighted'))
  acc_scores_test = np.append(acc_scores_test, accuracy_score(y_test_encoded, y_pred_test))
  f1_scores_test = np.append(f1_scores_test, f1_score(y_test_encoded, y_pred_test, average='weighted'))
fig=plt.figure()
# fig.set_size_inches((15, 10))
fig.show()
ax = fig.add_subplot(111)
ax.set_xticks(alg_nums)
ax.set_xlabel('Estimators num', fontdict={'fontsize': 15})
ax.set_ylabel('Score', fontdict={'fontsize': 15})

params = {'legend.fontsize': 14,
          'figure.figsize': (15, 10),
        }
plt.rcParams.update(params)


ax.plot(alg_nums, acc_scores_train, label='TRAIN ACCURACY')
ax.plot(alg_nums, f1_scores_train, label='TRAIN F1')
ax.plot(alg_nums, acc_scores_test, label='TEST ACCURACY')
ax.plot(alg_nums, f1_scores_test, label='TEST F1')


plt.title('Score plots w.r.t. logreg ensemble size', fontdict={'fontsize': 20})
plt.legend(loc='lower right');
from sklearn.ensemble import RandomForestClassifier
acc_scores_train = np.array([])
f1_scores_train = np.array([])
acc_scores_test = np.array([])
f1_scores_test = np.array([])

for n_estimators in tqdm(np.arange(n_min, n_max, step)):
  pipe = Pipeline([('scaler', StandardScaler()), 
                  ('dim_reduction', PCA(n_components=12)), 
                  ('clf', RandomForestClassifier(n_estimators=n_estimators))
                  ])

  pipe.fit(X_train, y_train_encoded)

  y_pred_test = pipe.predict(X_test)
  y_pred_train = pipe.predict(X_train)

  acc_scores_train = np.append(acc_scores_train, accuracy_score(y_train_encoded, y_pred_train))
  f1_scores_train = np.append(f1_scores_train, f1_score(y_train_encoded, y_pred_train, average='weighted'))
  acc_scores_test = np.append(acc_scores_test, accuracy_score(y_test_encoded, y_pred_test))
  f1_scores_test = np.append(f1_scores_test, f1_score(y_test_encoded, y_pred_test, average='weighted'))
fig=plt.figure()
# fig.set_size_inches((15, 10))
fig.show()
ax = fig.add_subplot(111)
ax.set_xticks(alg_nums)
ax.set_xlabel('Estimators num', fontdict={'fontsize': 15})
ax.set_ylabel('Score', fontdict={'fontsize': 15})

params = {'legend.fontsize': 14,
          'figure.figsize': (15, 10),
        }
plt.rcParams.update(params)


ax.plot(alg_nums, acc_scores_train, label='TRAIN ACCURACY')
ax.plot(alg_nums, f1_scores_train, label='TRAIN F1')
ax.plot(alg_nums, acc_scores_test, label='TEST ACCURACY')
ax.plot(alg_nums, f1_scores_test, label='TEST F1')


plt.title('Score plots w.r.t. tree ensemble size', fontdict={'fontsize': 20})
plt.legend(loc='lower right');
x_train_parts = np.array_split(X_train, 10)
y_train_parts = np.array_split(y_train_encoded, 10)

lr_acc = np.array([])
lr_f = np.array([])
tree_acc = np.array([])
tree_f = np.array([])
rf_acc = np.array([])
rf_f = np.array([])

acc_list = [lr_acc, tree_acc, rf_acc]
f_list = [lr_f, tree_f, rf_f]
models = [LogisticRegression(multi_class='multinomial', solver='saga', tol=1e-3, 
                             C=0.46415888336127775, penalty='l1'), 
          DecisionTreeClassifier(max_depth=7), 
          RandomForestClassifier(n_estimators=83)]

def get_model_scores(model, i, acc, f):
  pipe = Pipeline([('scaler', StandardScaler()), 
                 ('dim_reduction', PCA(n_components=12)), 
                 ('clf', model)])
  
  pipe.fit(np.concatenate(x_train_parts[:i + 1]), np.concatenate(y_train_parts[:i + 1]))
  y_pred_test = pipe.predict(X_test)

  acc = np.append(acc, accuracy_score(y_test_encoded, y_pred_test))
  f = np.append(f, f1_score(y_test_encoded, y_pred_test, average='weighted'))
  return acc, f

for i in tqdm(range(10)):
  for j in range(3):
    acc_list[j], f_list[j] = get_model_scores(models[j], i, acc_list[j], f_list[j])
fig=plt.figure()
# fig.set_size_inches((15, 10))
fig.show()
ax = fig.add_subplot(111)
# ax.set_xticks(alg_nums)
ax.set_xlabel('Data size', fontdict={'fontsize': 15})
ax.set_ylabel('Score', fontdict={'fontsize': 15})

params = {'legend.fontsize': 14,
          'figure.figsize': (15, 10),
        }
plt.rcParams.update(params)


ax.plot(np.arange(1, 11), acc_list[0], label='LOGREG ACCURACY')
ax.plot(np.arange(1, 11), f_list[0], label='LOGREG F1')
ax.plot(np.arange(1, 11), acc_list[1], label='TREE ACCURACY')
ax.plot(np.arange(1, 11), f_list[1], label='TREE F1')
ax.plot(np.arange(1, 11), acc_list[2], label='FOREST ACCURACY')
ax.plot(np.arange(1, 11), f_list[2], label='FOREST F1')


plt.title('Score plots w.r.t. data size', fontdict={'fontsize': 20})
plt.legend(loc='lower right');

























