# -*- coding: utf-8 -*-
"""
# Wine quality predictions with machine learning

### Authors:
Armin Shafiee, Fernando Moyano, Loay Ahmed Elalfy Abdelhafiz, Alexander Wemhoff

## Setup
"""

# Commented out IPython magic to ensure Python compatibility.
# -*- coding: utf-8 -*-

# Common imports
import time
import sys
import os
import urllib.request
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
# %matplotlib inline


# Setup of directories and options
ROOT_DIR = "."
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
DATA_DIR = os.path.join(ROOT_DIR, "data")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_raw_filepath = os.path.join(DATA_DIR, "winequality-red_raw.csv")
wine_filepath = os.path.join(DATA_DIR, "winequality-red.csv")

make_plots = False
GridSearch = False

# %matplotlib inline


# Some useful functions ----

# Save figures function
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_DIR, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# Download data function
def fetch_data(data_url, file_path):
    if os.path.isfile(file_path):
        print('File already exists.')
    else:
        urllib.request.urlretrieve(data_url, file_path)


# Recode csv file function
def recode_data(filein, fileout):
    fString = open(filein, "r")
    fFloat = open(fileout, "w")
    for line in fString:
        line = line.replace(";", ",")
        # line = line.replace("\t", ",")
        # line = line.replace("\r\n", "\n")
        fFloat.write(line)
    fString.close()
    fFloat.close()

"""## Data aquisition and exploration"""

# Get the data ----
fetch_data(WINE_URL, wine_raw_filepath)
recode_data(wine_raw_filepath, wine_filepath)
wine = pd.read_csv(wine_filepath)
# Change names for better plotting
wine.columns = ['fix_acid', 'volat_acid', 'citric_acid', 'resid_sugar',
                'chlorides', 'free_sulf_diox', 'tot_sulf_diox', 'density',
                'pH', 'sulphates', 'alcohol', 'quality']

# Display descriptive info and statistics ----
print(wine.head())
print(wine.info())
print(wine.describe())

# Create histograms to check data distributions ----
if make_plots:
    wine.hist(bins=50, figsize=(20, 15))
    save_fig("attribute_histogram_plots")
    plt.show()

# Correlations ----
corr_matrix = wine.corr()
corr_matrix['quality'].sort_values(ascending=False)

# Create a scatter matrix to check for correlations between variables
# Removed weaker correlations variables: 'resid_sugar', 'chlorides', 'free_sulf_diox', 'tot_sulf_diox', 'quality'
if make_plots:
    attributes = ['fix_acid', 'volat_acid', 'citric_acid', 'density', 'pH', 'sulphates', 'alcohol']
    scatter_matrix(wine[attributes], figsize=(28, 28))
    save_fig("scatter_matrix_plot")

"""## Data Preparation"""

# Split into features and target variables
X = wine.iloc[:, 0:11]
y = wine.iloc[:, 11]
features = X.columns

# Split into training and testing by stratifying using the target variable ('quality')
# Necessary because of very few low and high quality samples
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Testing scaling options
# # Normalizing lognormal data: option set to also standardize to 0 mean and unit sd
# # https://scikit-learn.org/stable/modules/preprocessing.html
powertrans = PowerTransformer(method='yeo-johnson', standardize=True)
X_test = powertrans.fit_transform(X_trn)

# Apply initial PCA to explore components
pca = PCA(n_components=11)
pca.fit(X_test)
pca_evr = pca.explained_variance_ratio_  # The first two components explain most of the variance.
pca_ev = pca.explained_variance_

# Plot cumulative sums of explained variance to decide on number of components
if make_plots:
    plt.subplot(121)
    plt.plot(np.concatenate(([0], np.cumsum(pca_evr))))
    plt.subplot(122)
    plt.plot(np.concatenate(([0], np.cumsum(pca_ev))))
    save_fig("expvar_pca")

# Making a Pipeline for the preprocessing
# We use 6 components for PCA to keep about 80% of the explained variance.

# Only scaling pipeline
s_pipeline = Pipeline([
        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True)),
    ])

# Only PCA pipeline
p_pipeline = Pipeline([
        ('pca', PCA(n_components=6)),
    ])

# Scaling a PCA pipeline
sp_pipeline = Pipeline([
        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True)),
        ('pca', PCA(n_components=6))
    ])

# Fit and transform on train data
X_trn_s = s_pipeline.fit_transform(X_trn)
X_trn_p = p_pipeline.fit_transform(X_trn)
X_trn_sp = sp_pipeline.fit_transform(X_trn)
# Transform only on test data
X_tst_s = s_pipeline.transform(X_tst)
X_tst_p = p_pipeline.transform(X_tst)
X_tst_sp = sp_pipeline.transform(X_tst)

X_list = list(zip(["X_trn", "X_trn_s", "X_trn_sp"], [X_trn, X_trn_s, X_trn_sp]))

# Look at distributions after scaling
if make_plots:
    pd.DataFrame(X_trn_s).hist(bins=50, figsize=(20, 15))
    save_fig("xtrain_histogram_plots")
    plt.show()

"""## LogisticRegression"""

log_reg = LogisticRegression(tol=0.0001, C=0.1, solver='newton-cg')
for names, X_pp in X_list:
  log_reg.fit(X_pp, y_trn)
  log_score = log_reg.score(X_pp, y_trn)
  log_cross_scores = np.mean(cross_val_score(log_reg, X_pp, y_trn, cv=7))
  print(names, log_score, log_cross_scores)

# Using best X dataset
log_reg.fit(X_trn_s, y_trn)

best_logreg_clf = log_reg

"""## K Nearest Neighbor"""

for i in range(1,9):
  for names, X_pp in X_list:
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_pp, y_trn)
    knn_score = neigh.score(X_pp, y_trn)
    knn_cross_scores = np.mean(cross_val_score(neigh, X_pp, y_trn, cv=7))
    print(names, i, knn_score, knn_cross_scores)

param_grid = [{'n_neighbors': [2,3,4,5,6,7,8], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}]
neigh = KNeighborsClassifier()
grid_search = GridSearchCV(neigh, param_grid, cv=7, return_train_score=True)
grid_search.fit(X_trn_s, y_trn)

# KNN with best parameters from grid search: K=5 kein PCA
neigh_gs = KNeighborsClassifier(n_neighbors=5)
neigh_gs.fit(X_trn_s, y_trn)
knn_score = neigh_gs.score(X_trn_s, y_trn)
knn_cross_scores = np.mean(cross_val_score(neigh_gs, X_trn_s, y_trn, cv=7))
print(knn_score,knn_cross_scores)

best_knn_clf = neigh_gs

"""## Support Vector Machine"""

# First test with default options
svc = SVC()

# Fit on different X datasets
for X_name, X in X_list:
    svc.fit(X, y_trn)
    print(f"Initial test score for {X_name}: {svc.score(X, y_trn)}")

# Cross validation on different datasets
for X_name, X in X_list:
    svc_scores = cross_val_score(svc, X, y_trn, cv=4, scoring="accuracy")
    print(f"Cross validation score mean for {X_name}: {np.mean(svc_scores)}")

# Best scores with scaled values: X_trn_s

if GridSearch:
    # Grid search
    param_grid = {'C': [0.1, 1, 10],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['rbf', 'poly', 'sigmoid'],
                  'decision_function_shape': ['ovo', 'ovr'],}

    svc_grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1)
    svc_grid.fit(X_trn_s, y_trn)
    print("Best Hyper Parameters for SVC grid search:\n", svc_grid.best_params_)

    # Cross validate using best parameter set
    svc_gs = SVC(C=1, gamma='scale', kernel='rbf', decision_function_shape='ovo')
    svc_scores = cross_val_score(svc_gs, X_trn_s, y_trn, cv=4, scoring="accuracy")
    print(f"Cross validation score mean for X_trn_s: {np.mean(svc_scores)}")

# No improvement in score after grid search. Using default.
svc = SVC(probability=True)
svc.fit(X_trn_s, y_trn)
best_svc_clf = svc

"""## Multilayer Perceptron (Neural Network Estimator)"""

# Initial test using 3 hidden layers and neurons = number of features ----

# Degrees of freedom with a maximum of 3 hidden layers
nl1 = 10
nl2 = 10
nl3 = 10
print(f"deg_freedom = {7 * nl1 + nl1 * nl2 + nl2 * nl3 + nl3 * 1}")

mlp_init = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500,
                    learning_rate_init=0.01, learning_rate='adaptive', random_state=2)

# Fit on different X datasets
for X_name, X in X_list:
    mlp_init.fit(X, y_trn)
    print(f"Initial test score for {X_name}: {mlp_init.score(X, y_trn)}")

# Common problem: optimization does not converge. Address later with grid search.

for X_name, X in X_list:
    mlp_scores = cross_val_score(mlp_init, X, y_trn, cv=4, scoring="accuracy")
    print(f"Initial test cross validation score mean for {X_name}: {np.mean(mlp_scores)}")

# Grid search for best parameters (using only X_trn_s)
if GridSearch:
    param_grid = [
            {
                'activation': ['tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'learning_rate': ['invscaling', 'adaptive'],
                'hidden_layer_sizes': [(10, 10), (20, 20), (30, 30), (30, 10), (10, 10, 10), (30, 20, 10)],
                'learning_rate_init': [0.01, 0.1],
                'max_iter': [300]
            }
          ]
    mlp = MLPClassifier(random_state=1)
    start = time.time()
    mlp_grid = GridSearchCV(mlp, param_grid, cv=4, scoring='accuracy')
    grid_result = mlp_grid.fit(X_trn_s, y_trn)
    end = time.time()
    print(f"Elapsed time: {end - start}")
    print(grid_result)
    print("Best Hyper Parameters:\n", mlp_grid.best_params_)

# Cross validation with best parameters of first grid search
mlp_gs = MLPClassifier(hidden_layer_sizes=(20, 20), activation='tanh',
                         learning_rate_init = 0.01, learning_rate='invscaling',
                         solver='adam', random_state=1, max_iter=1000)
mlp_gs.fit(X_trn_s, y_trn)
print(f"Score after using best pars of grid search: {mlp_gs.score(X_trn_s, y_trn)}")

mlp_gs_scores = cross_val_score(mlp_gs, X_trn_s, y_trn, cv=4, scoring="accuracy")
print(f"Mean cross validation score using best pars of grid search: {np.mean(mlp_gs_scores)}")

best_mlp_clf = mlp_gs

"""## DecisionTreeClassifier"""

L1 = ["X_trn", "X_trn_s"]
L2 = [X_trn, X_trn_s]
L12 = list(zip(L1, L2))
L3 = ["X_trn_p", "X_trn_sp"]
L4 = [X_trn_p, X_trn_sp]
L34 = list(zip(L3, L4))
L5 = L1 + L3
L6 = L2 + L4
L7 = [X_tst, X_tst_s, X_tst_p, X_tst_sp]
L8 = []
L9 = []
L10 = []
L_tree_test_scores = []
L_forest_test_scores = []


# def display_scores(scores):
    # print("Scores:", scores)
    # print("Mean training score:", scores.mean())
    # print("Standard deviation:", scores.std())

"""******************** DecisionTreeClassifier ********************"""

# """DecisionTreeClassifier without GridSearchCV"""
# L567 = list(zip(L5, L6, L7))
# for i, j, k in L567:
#     tree_clf = DecisionTreeClassifier(random_state=42)
#     tree_clf.fit(j, y_trn)
#     # y_pred = tree_clf.predict(j)
#     # tree_mse = mean_squared_error(y_trn, y_pred)
#     # tree_rmse = np.sqrt(tree_mse)
#     # print(tree_rmse)
#     tree_mse_scores = cross_val_score(tree_clf, j, y_trn,
#                              scoring="neg_mean_squared_error", cv=8)
#     tree_rmse_scores = np.sqrt(-tree_mse_scores)
#     # display_scores(tree_rmse_scores)
#     print(f"Mean tree training score for {i}: {tree_rmse_scores.mean()}")
#     tree_test_score = tree_clf.fit(j, y_trn).score(k, y_tst)
#     print(f"Tree test score for {i}: {tree_test_score}")

"""GridSearchCV for tree without PCA"""
if GridSearch:
    for i, j in L12:
        param_grid = [
            # try 10 hyperparameters
            {'max_features': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}
          ]
        tree_clf = DecisionTreeClassifier(random_state=42)
        # train across 8 folds, that's a total of 10*8=80 rounds of training
        grid_search = GridSearchCV(tree_clf, param_grid, cv=8,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(j, y_trn)
        L8.append(grid_search.best_params_['max_features'])
        print(f"The best tree parameters for {i}: {grid_search.best_estimator_}")

    """GridSearchCV for tree with PCA"""
    for i, j in L34:
        param_grid = [
            # try 5 hyperparameters
            {'max_features': [2, 3, 4, 5, 6]}
          ]
        tree_clf = DecisionTreeClassifier(random_state=42)
        # train across 8 folds, that's a total of 5*8=40 rounds of training
        grid_search = GridSearchCV(tree_clf, param_grid, cv=8,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(j, y_trn)
        L8.append(grid_search.best_params_['max_features'])
        print(f"The best tree parameters for {i}: {grid_search.best_estimator_}")

    """DecisionTreeClassifier after GridSearchCV for all X-variations"""
    L5678 = list(zip(L5, L6, L7, L8))
    for i, j, k, l in L5678:
        tree_clf = DecisionTreeClassifier(max_features=l, random_state=42)
        tree_clf.fit(j, y_trn)
        # y_pred = tree_clf.predict(j)
        # tree_mse = mean_squared_error(y_trn, y_pred)
        # tree_rmse = np.sqrt(tree_mse)
        # print(tree_rmse)
        tree_mse_scores = cross_val_score(tree_clf, j, y_trn,
                                 scoring="neg_mean_squared_error", cv=8)
        tree_rmse_scores = np.sqrt(-tree_mse_scores)
        # display_scores(tree_rmse_scores)
        print(f"\nMean tree training score for {i}: {tree_rmse_scores.mean()}")
        tree_test_score = tree_clf.fit(j, y_trn).score(k, y_tst)
        L_tree_test_scores.append(tree_test_score)
        print(f"Tree test score for {i}: {tree_test_score}")

    """DecisionTreeClassifier with the best parameters for X_trn_s (automated version)"""
    best_tree_clf = DecisionTreeClassifier(max_features=L8[1], random_state=42)
    best_tree_clf.fit(L6[1], y_trn)
else:    
    """DecisionTreeClassifier with the best parameters for X_trn_s (manual version)"""
    best_tree_clf = DecisionTreeClassifier(max_features=4, random_state=42)
    best_tree_clf.fit(L6[1], y_trn)

"""## RandomForestClassifier"""

"""******************** RandomForestClassifier ********************"""

# """RandomForestClassifier without GridSearchCV"""
# L567 = list(zip(L5, L6, L7))
# for i, j, k in L567:
#     forest_clf = RandomForestClassifier(random_state=42)
#     forest_clf.fit(j, y_trn)
#     # y_pred = forest_clf.predict(j)
#     # forest_mse = mean_squared_error(y_trn, y_pred)
#     # forest_rmse = np.sqrt(forest_mse)
#     # print(forest_rmse)
#     forest_mse_scores = cross_val_score(forest_clf, j, y_trn,
#                              scoring="neg_mean_squared_error", cv=8)
#     forest_rmse_scores = np.sqrt(-forest_mse_scores)
#     # display_scores(tree_rmse_scores)
#     print(f"Mean forest training score for {i}: {forest_rmse_scores.mean()}")
#     forest_test_score = forest_clf.fit(j, y_trn).score(k, y_tst)
#     print(f"Forest test score for {i}: {forest_test_score}")

"""GridSearchCV for forest without PCA"""
if GridSearch:
    for i, j in L12:
        param_grid = [
            # try 15 (3×5) combinations of hyperparameters
            {'n_estimators': [50, 100, 150], 'max_features': [3, 5, 7, 9, 11]},
            # then try 15 (3×5) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [50, 100, 150], 'max_features': [3, 5, 7, 9, 11]},
          ]
        forest_clf = RandomForestClassifier(random_state=42)
        # train across 8 folds, that's a total of (15+15)*8=240 rounds of training
        grid_search = GridSearchCV(forest_clf, param_grid, cv=8,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(j, y_trn)
        L9.append(grid_search.best_params_['n_estimators'])
        L10.append(grid_search.best_params_['max_features'])
        print(f"The best forest parameters for {i}: {grid_search.best_estimator_}")

    """GridSearchCV for forest with PCA"""
    for i, j in L34:
        param_grid = [
            # try 15 (3×5) combinations of hyperparameters
            {'n_estimators': [50, 100, 150], 'max_features': [2, 3, 4, 5, 6]},
            # then try 15 (3×5) combinations with bootstrap set as False
            {'bootstrap': [False], 'n_estimators': [50, 100, 150], 'max_features': [2, 3, 4, 5, 6]},
        ]
        forest_clf = RandomForestClassifier(random_state=42)
        # train across 8 folds, that's a total of (15+15)*8=240 rounds of training
        grid_search = GridSearchCV(forest_clf, param_grid, cv=8,
                                   scoring='neg_mean_squared_error',
                                   return_train_score=True)
        grid_search.fit(j, y_trn)
        L9.append(grid_search.best_params_['n_estimators'])
        L10.append(grid_search.best_params_['max_features'])
        print(f"The best forest parameters for {i}: {grid_search.best_estimator_}")

    """RandomForestClassifier after GridSearchCV for all X-variations"""
    L567910 = list(zip(L5, L6, L7, L9, L10))
    for i, j, k, l, m in L567910:
        forest_clf = RandomForestClassifier(n_estimators=l, max_features=m, random_state=42)
        forest_clf.fit(j, y_trn)
        # y_pred = forest_clf.predict(j)
        # forest_mse = mean_squared_error(y_trn, y_pred)
        # forest_rmse = np.sqrt(forest_mse)
        # print(forest_rmse)
        forest_mse_scores = cross_val_score(forest_clf, j, y_trn,
                                 scoring="neg_mean_squared_error", cv=8)
        forest_rmse_scores = np.sqrt(-forest_mse_scores)
        # display_scores(tree_rmse_scores)
        print(f"\nMean forest training score for {i}: {forest_rmse_scores.mean()}")
        forest_test_score = forest_clf.fit(j, y_trn).score(k, y_tst)
        L_forest_test_scores.append(forest_test_score)
        print(f"Forest test score for {i}: {forest_test_score}")

    """RandomForestClassifier with the best parameters for X_trn_s (automated version)"""
    best_forest_clf = RandomForestClassifier(n_estimators=L9[1], max_features=L10[1], random_state=42)
    best_forest_clf.fit(L6[1], y_trn)
else:
    """RandomForestClassifier with the best parameters for X_trn_s (manual version)"""
    best_forest_clf = RandomForestClassifier(n_estimators=150, max_features=5, random_state=42)
    best_forest_clf.fit(L6[1], y_trn)

"""## Ensemble Voting"""

# Ensemble voting ----
eclfh = VotingClassifier(estimators=[('knn', best_knn_clf), ('mlp', best_mlp_clf), ('dt', best_tree_clf), ('rf', best_forest_clf), ('svc', best_svc_clf), ('lr', best_logreg_clf)], voting='hard')
eclfh.fit(X_trn_s, y_trn)

eclfs = VotingClassifier(estimators=[('knn', best_knn_clf), ('mlp', best_mlp_clf), ('dt', best_tree_clf), ('rf', best_forest_clf), ('svc', best_svc_clf), ('lr', best_logreg_clf)], voting='soft')
eclfs.fit(X_trn_s, y_trn)

"""## Final scoring with test data"""

# MLP scores
# mlp_init.fit(X_trn_s, y_trn)
# test_score = mlp_init.score(X_tst_s, y_tst)
# print(f"Score for MLP (init): {test_score}")
test_score = best_mlp_clf.score(X_tst_s, y_tst)
print(f"Score for MLP: {test_score}")

# KNN scores
test_score = best_knn_clf.score(X_tst_s, y_tst)
print(f"Score for KNN: {test_score}")

# LogReg scores
test_score = best_logreg_clf.score(X_tst_s, y_tst)
print(f"Score LogReg: {test_score}")

# Decision Tree scores
test_score = best_tree_clf.score(X_tst_s, y_tst)
print(f"Score for Decision Tree: {test_score}")

# Random Forest Classifier
test_score = best_forest_clf.score(X_tst_s, y_tst)
print(f"Score for Random Forest: {test_score}")

# SVC
test_score = best_svc_clf.score(X_tst_s, y_tst)
print(f"Score for SVC: {test_score}")

# Ensemble voting
test_score = eclfh.score(X_tst_s, y_tst)
print(f"Score for hard voting classifier: {test_score}")

test_score = eclfs.score(X_tst_s, y_tst)
print(f"Score for soft voting classifier: {test_score}")

"""## Agglomerative clustering"""

D = pdist(X_trn_s, metric='euclidean')
Z = linkage(D, 'average', metric='euclidean')
#clustering
clus = fcluster(Z,6,criterion='maxclust')+3
validation=y_trn-clus
vd = pd.DataFrame(validation)
print(vd.value_counts())