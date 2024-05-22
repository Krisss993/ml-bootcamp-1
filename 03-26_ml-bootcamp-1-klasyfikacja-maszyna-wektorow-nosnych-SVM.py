import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import plotly.offline as pyo

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.datasets import load_iris

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus


from sklearn.datasets import make_regression
import statsmodels.api as sm


from sklearn.svm import SVC


sns.set()

from mlxtend.plotting import plot_decision_regions


from scipy.stats import entropy


from sklearn.ensemble import RandomForestClassifier


raw_data = load_iris()
all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']
feature_names = all_data['feature_names']
target_names = all_data['target_names']

df = pd.DataFrame(data=np.c_[data,target], columns=feature_names+['target'])
# df = df[(df['target'] == 0) | (df['target'] == 1)]
df

data = df.iloc[:,[1,2]].values
target = df['target'].astype('int').values

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)

# STANDARYZUJEM DANE TRENINGOWE JAK I TESTOWE
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


classifier = SVC(C=1.0, kernel='linear')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
classifier.score(X_test, y_test)


plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, classifier)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f' max_depth=1, accuracy: {classifier.score(X_test, y_test)}%')
plt.show()








classifier = SVC(C=1.0, kernel='rbf')
classifier.fit(X_train, y_train)

plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, classifier)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f' max_depth=1, accuracy: {classifier.score(X_test, y_test)}%')
plt.show()







classifier = SVC(C=1.0, kernel='poly')
classifier.fit(X_train, y_train)

plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, classifier)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f' max_depth=1, accuracy: {classifier.score(X_test, y_test)}%')
plt.show()







