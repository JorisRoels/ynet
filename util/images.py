# this scripts generates the images from the paper

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# constants
# domain names
EPFL = 'EPFL'
UROCELL = 'UroCell'
PO936Q = 'Po936Q'
MIRA = 'MiRA'
KASTHURI = 'Kasthuri'
VNC = 'VNC'
EMBL_HELA = 'EMBL-HeLa'
VIB_EVHELA = 'VIB-evHeLa'
VIB_SHPERK = 'VIB-shPerk'
# class labels
MITO = 'M'
ER = 'ER'
NM = 'NM'
# class colors
MITO_COLOR = 'tab:red'
ER_COLOR = 'tab:green'
NM_COLOR = 'tab:blue'


def load_results(basepath, domains, n_exp, metrics, class_name):
    df_final = pd.DataFrame(columns=['measurement', 'metric', 'domain', 'experiment', 'class'])
    df_best = pd.DataFrame(columns=['measurement', 'metric', 'domain', 'experiment', 'class'])
    for i, dom in enumerate(domains):
        for n in range(n_exp):
            r_final = np.load(os.path.join(basepath, dom, str(n), 'validation_final.npy'))
            r_best = np.load(os.path.join(basepath, dom, str(n), 'validation_best.npy'))
            for j, metric in enumerate(metrics):
                df_final = df_final.append(
                    {'measurement': r_final.mean(axis=0)[j], 'metric': metric, 'domain': dom, 'experiment': n,
                     'class': class_name}, ignore_index=True)
                df_best = df_best.append(
                    {'measurement': r_best.mean(axis=0)[j], 'metric': metric, 'domain': dom, 'experiment': n,
                     'class': class_name}, ignore_index=True)
    return df_final, df_best


# Figure 1. Full-TD performance
full_td_img = '../train/exp/results/full-target-supervision/figure-1.pdf'
# data
domains = [EPFL, UROCELL, PO936Q, MIRA, KASTHURI, VNC]
n_exp = 5
metrics = ['mIoU', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'F1 score']
basepath = '../train/exp/results/full-target-supervision'
_, df_best_mito = load_results(basepath, domains, n_exp, metrics, MITO)

# plot everything
colors = 6 * [MITO_COLOR] + 3 * [ER_COLOR] + 2 * [NM_COLOR]
domains = [EPFL, UROCELL, PO936Q, MIRA, KASTHURI, VNC, EMBL_HELA, VIB_EVHELA, VIB_SHPERK, VIB_EVHELA, VIB_SHPERK]
sns.set_style('whitegrid')
ax = sns.barplot(data=df_best_mito[df_best_mito['metric'] == 'mIoU'], y='domain', x='measurement', palette=colors)
# make it look nicer
plt.legend((MITO, ER, NM), loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
leg = ax.get_legend()
leg.legendHandles[0].set_color(MITO_COLOR)
leg.legendHandles[1].set_color(ER_COLOR)
leg.legendHandles[2].set_color(NM_COLOR)
plt.xlabel('mIoU')
plt.xlim([0.6, 1.0])
locs, labels = plt.yticks()
plt.yticks(locs, domains)
plt.ylabel(None)
plt.tight_layout()
# save file
plt.savefig(full_td_img, format='pdf')
