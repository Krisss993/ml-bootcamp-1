
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
import plotly.express as px
import plotly.offline as pyo
import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.plotting import plot_decision_regions

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import datasets

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets.fashion_mnist import load_data


#########################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
#########################



np.random.seed(42)
sns.set(font_scale=1.3)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=1000, formatter=dict(float=lambda x: f'{x:.2f}'))



documents = [
    'Today is Friday',
    'I like Friday',
    'Today I am going to learn Python.',
    'Friday, Friday!!!'
]

print(documents)
     

vectorizer = CountVectorizer()
vectorizer.fit_transform(documents)


# toarray PODAJE LICZEBNOSC KAZDEGO SLOWA W POJEDYNCZYM DOKUMENCIE Z LISTY documents, SLOWA TE WYSWIETLA get_feature_names_out
vectorizer.fit_transform(documents).toarray()
vectorizer.get_feature_names_out()

# DF WYSWIETLAJACY DANE POWYZEJ
df = pd.DataFrame(data = vectorizer.fit_transform(documents).toarray(), columns=vectorizer.get_feature_names_out())
df

# PODAJE INDEKS KAZDEGO SLOWA
vectorizer.vocabulary_

# UZYWAMY TRANSFORM, PONIEWAZ CHCEMY DOPASOWAC NOWE DANE DO TYCH KTORE JUZ NASZ MODEL ZNA
# ZAKODOWANE ZOSTANIE TYLKO Friday PONIEWAZ TYLKO TO POZNAL MODEL
vectorizer.transform(['Friday morning']).toarray()




# DOSTOWOWUJE PARY WYRAZOW, JEST TO UZYTECZNE ZE WZGLEDU NA ORTOGRAFIE (WYSTEPOWANIE WYRAZOW OBOK SIEBIE)
bigram = CountVectorizer(ngram_range=(1,2), min_df=1) # min_df=2
bigram.fit_transform(documents).toarray()

df = pd.DataFrame(data = bigram.fit_transform(documents).toarray(), columns=bigram.get_feature_names_out())
df

# PODAJE INDEKSY BIGRAMOW I POJEDYNCZYCH SLOW
bigram.vocabulary_








# TFID TRANSFORMER
documents = [
    'Friday morning',
    'Friday chill',
    'Friday - morning',
    'Friday, Friday morning!!!'
]

print(documents)

counts = vectorizer.fit_transform(documents).toarray()
counts

df = pd.DataFrame(data=vectorizer.fit_transform(documents).toarray(), columns=vectorizer.get_feature_names_out())
df

# DAJE WAGI(WAZNOSC) SLOW W DOKUMENCIE, WYROZNIA SLOWA UNIKATOWE W KAZDYM DOKUMENCIE, IM RZADZIEJ SLOWO WYSTEPUJE W KAZDYM Z DOKUMENTOW TYM WIEKSZA MA WAGE
# NP JESLI SLOWO WYSTEPUJE W JEDNYM DOKUMENCIE MA WIEKSZA WAGE NIZ TO CO WYSTEPUJE W KILKU
# WYKRYWA SLOWA SPECYFICZNE DLA DANEGO TYPU DOKUMENTOW, MNIEJSZA WAGE NADAJE SLOWOM GENERYCZNYM
tfidf = TfidfTransformer()
tfidf.fit_transform(counts).toarray()



# TFIDF VECTORIZER


tfidf_vec = TfidfVectorizer()
tfidf_vec.fit_transform(documents).toarray()


raw_data = fetch_20newsgroups(subset='train', categories=['comp.graphics'], random_state=42)
all_data = raw_data.copy()
all_data['data'][:5]


all_data['target_names']

all_data['target'][:10]


tfidf = TfidfVectorizer()
tfidf.fit_transform(all_data['data']).toarray()


























# MODEL KLASYFIKACJI RECENZJI FILMOWYCH NA POZYTYWNE I NEGATYWNE






raw_movie = load_files('movie_reviews')
movie = raw_movie.copy()
movie.keys()

movie['data'][:10]
movie['target_names']
movie['filenames'][:2]
movie['target']

X_train, X_test, y_train, y_test = train_test_split(movie['data'], movie['target'], random_state=42)



print(f'X_train: {len(X_train)}')
print(f'X_test: {len(X_test)}')



X_train[0]

tfidf = TfidfVectorizer(max_features=3000)
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')



classifier = MultinomialNB()
classifier.fit(X_train, y_train)
classifier.score(X_test, y_test)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

def conf_mat(cm):
    cm=cm[::-1]
    df = pd.DataFrame(data=cm, columns=movie['target_names'], index = movie['target_names'])
    fig = ff.create_annotated_heatmap(z=df.values, x=list(df.columns), y=list(df.index), colorscale='ice', reversescale=True, showscale=True)
    pyo.plot(fig)

conf_mat(cm)

print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))


new_reviews = ['It was awesome! Very interesting story.', 
               'I cannot recommend this film. Short and awful.',
               'Very long and boring. Don\'t waste your time.',
               'Well-organized and quite interesting.']

new_reviews_tfidf = tfidf.transform(new_reviews)
new_reviews_tfidf.toarray()

new_rev_pred = classifier.predict(new_reviews_tfidf)
new_rev_pred

new_rev_preob = classifier.predict_log_proba(new_reviews_tfidf)
new_rev_preob


