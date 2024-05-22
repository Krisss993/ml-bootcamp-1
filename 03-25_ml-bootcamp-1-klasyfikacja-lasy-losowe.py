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

sns.set()




from scipy.stats import entropy


from sklearn.ensemble import RandomForestClassifier

raw_data = load_iris()

feature_names = [name.replace(' ', '_')[:-5] for name in raw_data['feature_names']]

df = pd.DataFrame(data=np.c_[raw_data.data, raw_data.target], columns=feature_names+['target'])
df

data = df.copy()
data = data[['sepal_length', 'sepal_width', 'target']]
target = data.pop('target')




classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(data, target)

classifier.score(data, target)

data = data.values
target = target.values.astype('int16')

from mlxtend.plotting import plot_decision_regions

plt.figure(figsize=(8, 6))
plot_decision_regions(data, target, classifier)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f'Drzewo decyzyjne: max_depth=1, accuracy: {classifier.score(data, target) * 100:.2f}%')
plt.show()






data = raw_data['data']
targets =raw_data['target']
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(data, target)

# SPRAWDZAMY KTORE CECHY BYLY WAZNE DLA MODELU 
classifier.feature_importances_

features = pd.DataFrame(data={'feature': feature_names, 'feature_importance': classifier.feature_importances_})

fig = px.bar(features, x='feature', y='feature_importance', width=700, height=400)
pyo.plot(fig)

