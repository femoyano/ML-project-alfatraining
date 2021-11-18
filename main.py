# -*- coding: utf-8 -*-
"""
Some text that can be added to the report
1. Klärung der Aufgabenstellung
  1.a Was ist das Ziel?
    Ziel: predict the quality of a wine as a fucntion of its chemical properties.
  1.b Was sind die Qualitätskriterien an denen die Lösung gemessen wird?
    Amount of correctly predicted cases of wine quality
  1.c Ist dies eine Aufgabe für überwachtes Lernen oder für nicht-überwachtes Lernen?
    For supervised learning. (may later apply clustering to compare with predictive model results).
"""

# Code to remove if not needed. In any case, avoid copy-paste.
import sys
# assert sys.version_info >= (3, 5)  # remove?
# Scikit-Learn ≥0.20 is required  # remove?
# assert sklearn.__version__ >= "0.20"  # remove?
# import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)

# Common imports
import os
import urllib.request
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# %matplotlib inline

# ----------------
# SETUP ----------
# ----------------

ROOT_DIR = "."
IMAGES_DIR = os.path.join(ROOT_DIR, "images")
DATA_DIR = os.path.join(ROOT_DIR, "data")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_raw_filepath = os.path.join(DATA_DIR, "winequality-red_raw.csv")
wine_filepath = os.path.join(DATA_DIR, "winequality-red.csv")

make_plots = False

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

# Useful functions ----

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


# Get the data ----
fetch_data(WINE_URL, wine_raw_filepath)
recode_data(wine_raw_filepath, wine_filepath)
wine = pd.read_csv(wine_filepath)
# Change names for better plotting
wine.columns = ['fix_acid', 'volat_acid', 'citric_acid', 'resid_sugar',
                'chlorides', 'free_sulf_diox', 'tot_sulf_diox', 'density',
                'pH', 'sulphates', 'alcohol', 'quality']

# Display descriptive info and statistics ----
wine.head()
wine.info()
wine.describe()

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


# Data Preparation ----

# Split into features and target variables
X = wine.iloc[:, 0:11]
y = wine.iloc[:, 11]
features = X.columns

# Split into training and testing by stratifying using the target variable ('quality')
# Necessary because of very few low and high quality samples
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizing lognormal data: option set to also standardize to 0 mean and unit sd
# https://scikit-learn.org/stable/modules/preprocessing.html

if make_plots:
    pd.DataFrame(X_trn).hist(bins=50, figsize=(20, 15))
    save_fig("xtrain_histogram_plots")
    plt.show()

# Apply PCA
pca = PCA(n_components=11)
Xtrn1 = pca.fit_transform(X_trn)
pca_evr = pca.explained_variance_ratio_  # The first two components explain most of the variance.
pca_ev = pca.explained_variance_

# Plot cumulative sums of explained variance to decide on number of components
if make_plots:
    plt.subplot(121)
    plt.plot(np.concatenate(([0], np.cumsum(pca_evr))))
    plt.subplot(122)
    plt.plot(np.concatenate(([0], np.cumsum(pca_ev))))

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

X_list = list(zip(["X_trn", "X_trn_s", "X_trn_p", "X_trn_sp"], [X_trn, X_trn_s, X_trn_p, X_trn_sp]))
