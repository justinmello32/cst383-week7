import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_splitimport 
import graphviz

df = pd.read_csv("https://raw.githubusercontent.com/grbruns/cst383/master/heart.csv")
df['output'] = df['output'] - 1
df = df[['age', 'maxhr', 'restbp', 'output']]
sns.scatterplot(x='age', y='maxhr', hue='output', data=df)

#1
#Roughly .47

#2
def gini(class_counts):
    """ return the Gini value for a node in a binary classif. tree """
    if sum(class_counts) == 0:
        return 0
        p = class_counts[0]/sum(class_counts)
        return 2 * p * (1 - p)

#3
#Tested multiple values

#4
#Added additional code

#5
#In between 20 - 30?

#6
gini_root = gini([(df['output'] == i).sum() for i in [0,1]])

#7
split_val = 50
df_lo = df[df['age'] < split_val]
df_hi = df[df['age'] >= split_val]
counts_lo = [(df_lo['output'] == i).sum() for i in [0,1]]
counts_hi = [(df_hi['output'] == i).sum() for i in [0,1]]
gini_lo = gini(counts_lo)
gini_hi = gini(counts_hi)