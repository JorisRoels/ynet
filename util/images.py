# this scripts generates the images from the paper

import seaborn as sns
import matplotlib.pyplot as plt
import pickle

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
# results files
full_td_file = '../train/exp/results/full-target-supervision/results_best.pickle'
# save files
full_td_img = '../train/exp/results/full-target-supervision/figure.pdf'

# Figure 1. Full-TD performance
# data
with open(full_td_file, 'rb') as ftd_file:
    df_best = pickle.load(ftd_file)
# plot everything
colors = 6 * [MITO_COLOR] + 3 * [ER_COLOR] + 2 * [NM_COLOR]
domains = [EPFL, UROCELL, PO936Q, MIRA, KASTHURI, VNC, EMBL_HELA, VIB_EVHELA, VIB_SHPERK, VIB_EVHELA, VIB_SHPERK]
sns.set_style('whitegrid')
ax = sns.barplot(data=df_best[df_best['metric'] == 'mIoU'], y='domain', x='measurement', palette=colors)
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
plt.savefig(full_td_img, format='pdf')
