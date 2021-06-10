import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_splitimport 
import graphviz

#1
#Non-parametric method means that there are no underlying assumptions about the distribution of the errors or the data

# 2
df = pd.read_csv('https://raw.githubusercontent.com/grbruns/cst383/master/College.csv', index_col=0)

#3
df.loc[(df['Private'] == 'Yes'), 'Private'] = 1
df.loc[(df['Private'] == 'No'), 'Private'] = 0

#4
#Explored data using df.info() and df.describe(). Also viewed data using dataframe explorer.

#5
predictors = ['Outstate', 'F.Undergrad']
X = df[predictors].values
y = df['Private'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#7
dot_data = export_graphviz(clf, precision=2,feature_names=predictors,  proportion=True,class_names=target_names,  filled=True, rounded=True,  special_characters=True)
graph = graphviz.Source(dot_data)