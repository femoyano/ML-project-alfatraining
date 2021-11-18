"""
Code for setting up a multilayered perceptron (MPL) estimator.
Relevant links:
https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

"""
from main import *

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import time


gridsearch = False

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
    print(f"Initial test score {X_name}: {mlp_init.score(X, y_trn)}")

# Common problem: optimization does not converge. Address later with grid search.

for X_name, X in X_list:
    mlp_scores = cross_val_score(mlp_init, X, y_trn, cv=4, scoring="accuracy")
    print(f"Initial test cross validation score mean for {X_name}: {np.mean(mlp_scores)}")

# Grid search for best parameters (using only X_trn_s)
if gridsearch:
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

# Prediction and score using test data
mlp_init.fit(X_trn_s, y_trn)
mlp_test_score = mlp_init.score(X_tst_s, y_tst)
print(f"Score mlp_init: {mlp_test_score}")
mlp_test_score = mlp_gs.score(X_tst_s, y_tst)
print(f"Score mlp_gs: {mlp_test_score}")
