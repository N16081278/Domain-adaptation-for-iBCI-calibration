import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import itertools


sns.set_theme(style='whitegrid', font_scale=2)

df_mix = pd.read_csv('./result_experiments/experiments_mixCalibration.csv')
df_ft = pd.read_csv('./result_experiments/experiments_finetune.csv')

df = pd.concat((df_mix, df_ft), axis=0).reset_index(drop=True)
df = df[df['axis'] == 'x'].reset_index(drop=True)
print(df)


g = sns.catplot(kind='bar', data=df, x='targetSessionIndex', y='r_square', hue='model', \
    col='sourceSessionIndex', col_wrap=6, ci=None)


# plt.show()
plt.savefig('./a.jpg')