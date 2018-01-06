import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import SelectFromModel, SelectKBest, RFE, f_regression, chi2, f_classif
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from sklearn.cluster import KMeans, k_means
from sklearn import metrics
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
