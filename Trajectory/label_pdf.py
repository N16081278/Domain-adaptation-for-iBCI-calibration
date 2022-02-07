#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:08:52 2021

@author: ogk
"""

import get_data
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

path = '/home/ogk/Documents/OGK/Trajectory/vel_PDF/'

# session = 20
# bins = 50
# velocity = get_data.data_preprocess(session)

# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True) # constrained_layout=True
# time = 1 # mins
# duration_1 = int(time*(60*1000)/64)
# axs[0, 0].set_title('1 minute')
# axs[0, 0].hist(velocity[:duration_1, 0], bins=bins, density=True, color='#0055FF', ec='#0000FF', alpha=0.5)
# axs[0, 0].hist(velocity[:duration_1, 1], bins=bins, density=True, color='#FF8800', ec='#FF0000', alpha=0.5)
# # sns.distplot(velocity[:duration_1, 1], ax=axs[0, 0])
# axs[0, 0].legend(['V_x', 'V_y'])
# axs[0, 0].set_ylabel('Probability')

# time = 3 # mins
# duration_3 = int(time*(60*1000)/64)
# axs[0, 1].set_title('3 minutes')
# axs[0, 1].hist(velocity[:duration_3, 0], bins=bins, density=True, color='#0055FF', ec='#0000FF', alpha=0.5)
# axs[0, 1].hist(velocity[:duration_3, 1], bins=bins, density=True, color='#FF8800', ec='#FF0000', alpha=0.5)
# axs[0, 1].legend(['V_x', 'V_y'])

# time = 5 # mins
# duration_5 = int(time*(60*1000)/64)
# axs[1, 0].set_title('5 minutes')
# axs[1, 0].hist(velocity[:duration_5, 0], bins=bins, density=True, color='#0055FF', ec='#0000FF', alpha=0.5)
# axs[1, 0].hist(velocity[:duration_5, 1], bins=bins, density=True, color='#FF8800', ec='#FF0000', alpha=0.5)
# axs[1, 0].legend(['V_x', 'V_y'])
# axs[1, 0].set_ylabel('Probability')

# time = 1 # mins
# duration_1 = int(time*(60*1000)/64)
# axs[1, 1].set_title('else')
# axs[1, 1].hist(velocity[duration_5:, 0], bins=bins, density=True, color='#0055FF', ec='#0000FF', alpha=0.5)
# axs[1, 1].hist(velocity[duration_5:, 1], bins=bins, density=True, color='#FF8800', ec='#FF0000', alpha=0.5)
# axs[1, 1].legend(['V_x', 'V_y'])

# fig.tight_layout()


session = 1
bins = 50
velocity = get_data.data_preprocess(session)
time = 1 # mins
duration_1 = int(time*(60*1000)/64)
time = 3 # mins
duration_3 = int(time*(60*1000)/64)
time = 5 # mins
duration_5 = int(time*(60*1000)/64)

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True) # constrained_layout=True
kwargs = dict(bins=50)

axs[0].set_title('V_x')
# axs[0].hist(velocity[:duration_1, 0], **kwargs)
# axs[0].hist(velocity[:duration_3, 0], **kwargs)
# axs[0].hist(velocity[:duration_5, 0], **kwargs)
# axs[0].hist(velocity[duration_5:, 0], **kwargs)
sns.distplot(velocity[:duration_1, 0], ax=axs[0], color='b', **kwargs)
sns.distplot(velocity[:duration_3, 0], ax=axs[0], color='orange', **kwargs)
sns.distplot(velocity[:duration_5, 0], ax=axs[0], color='lime', **kwargs)
sns.distplot(velocity[duration_5:, 0], ax=axs[0], color='r', **kwargs)
axs[0].legend(['1 minute', '3 minute', '5 minutes', 'else'])
axs[0].set_ylabel('Probability')

axs[1].set_title('V_y')
sns.distplot(velocity[:duration_1, 1], ax=axs[1], color='b', **kwargs)
sns.distplot(velocity[:duration_3, 1], ax=axs[1], color='orange', **kwargs)
sns.distplot(velocity[:duration_5, 1], ax=axs[1], color='lime', **kwargs)
sns.distplot(velocity[duration_5:, 1], ax=axs[1], color='r', **kwargs)
# sns.distplot(velocity[:duration_1, 1], ax=axs[0, 0])
axs[1].legend(['1 minute', '3 minute', '5 minutes', 'else'])
axs[1].set_ylabel('Probability')
fig.tight_layout()