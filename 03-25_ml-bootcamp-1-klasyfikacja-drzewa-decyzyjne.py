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

entropy([0.5, 0.5], base=2)
entropy([1, 0], base=2)
entropy([0.8, 0.2], base=2)

def entrop(x):
    return -np.sum(x*np.log2(x))

entrop([0.5,0.5])











raw_data = load_iris()
all_data = raw_data.copy()

data = all_data['data']
target = all_data['target']
feature_names = [name.replace(' ', '_')[:-5] for name in all_data['feature_names']]
target_names = all_data['target_names']


df = pd.DataFrame(data=np.c_[data, target], columns=feature_names + ['target'])
df.head()


data = df.copy()
data = data[['sepal_length', 'sepal_width', 'target']]
target = data.pop('target')

data.head()



X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


classifier = DecisionTreeRegressor(max_depth=4)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

classifier.score(X_test, y_test)






data = data.values
target = target.values.astype('int16')



from mlxtend.plotting import plot_decision_regions

colors='#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'


acc = classifier.score(data, target)

plt.figure(figsize=(8, 6))
plot_decision_regions(data, target, classifier, colors=colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f'Drzewo decyzyjne: max_depth=1, accuracy: {acc * 100:.2f}%')
plt.show()


plt.figure(figsize=(12, 8), facecolor='lightgrey', )
plot_tree(classifier, filled=True, rounded=True, proportion=True, feature_names=feature_names[:2])
plt.show()
