from django.test import TestCase

# Create your tests here.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import random
from collections import Counter
plt.rcParams['font.sans-serif']=['SimHei']
low2low, low2normal, low2high = 0, 0, 0
normal2low, normal2normal, normal2high = 0, 0, 0
high2low, high2normal, high2high = 0, 0, 0
confusion_matrix=[[low2low, low2normal, low2high],
                  [normal2low, normal2normal, normal2high],
                  [high2low, high2normal, high2high]]
fig, ax = plt.subplots(figsize=(9, 9))
cmap = mcolors.LinearSegmentedColormap.from_list('n', ['#ffff99', '#ff0099'])
sns.heatmap(pd.DataFrame(confusion_matrix,
                         columns=['低体重儿', '正常体重儿', '巨大儿'],
                         index=['低体重儿', '正常体重儿', '巨大儿']), annot=True, fmt='d', vmax=50, vmin=0, cmap=cmap, square=True,
            linewidths=0.01, linecolor='white', cbar=False)
plt.gca().xaxis.set_ticks_position('top')
plt.xticks(size=12)
plt.yticks(size=12, rotation=0)
plt.savefig('ddemo.png')