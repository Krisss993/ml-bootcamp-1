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

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
import pydotplus
from sklearn import tree as tree_graph

from sklearn.datasets import make_regression
import statsmodels.api as sm

sns.set()

data, target = make_regression(n_samples=200, n_features=1, noise=20)
target = target ** 2

plot_data = np.arange(-3, 3, 0.01).reshape(-1, 1)

print(f'{data[:5]}\n')
print(target[:5])

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.scatter(data, target, label='dane')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
plt.show()
     

lr = LinearRegression()
lr.fit(data, target)
y_pred = lr.predict(plot_data)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.scatter(data, target, label='dane')
plt.plot(plot_data, y_pred, label='regresja liniowa', color='red')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
plt.show()

lr.score(data, target)



lr_pol = LinearRegression()
poly = PolynomialFeatures(degree=2)
data_poly = poly.fit_transform(data)


lr_pol.fit(data_poly, target)
yp_pred = lr_pol.predict(data_poly)

lr_pol.score(data_poly, target)


plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.scatter(data, target, label='dane')
plt.scatter(data, yp_pred, label='regresja wielomianowa', color='red')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
plt.show()


tree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)

tree.fit(data, target)
yt_pred = tree.predict(plot_data)

plt.figure(figsize=(8, 6))
plt.title('Regresja drzew decyzyjnych')
plt.plot(plot_data, yt_pred, label='regresja liniowa', color='red')
plt.scatter(data, target, label='dane')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
plt.show()



plt.figure(figsize=(12, 8), facecolor='lightgrey', )
plot_tree(tree, filled=True, feature_names=['cecha x'], rounded=True, proportion=True)
plt.show()


# NIE DZIALA W SPYDER
# dot_data = StringIO()
# export_graphviz(tree, out_file=dot_data,
#                 filled=True, rounded=True,
#                 special_characters=True,
#                 feature_names=['cecha x'])
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('graph.png')

# Image(graph.create_png(), width=600)
