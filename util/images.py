# this scripts generates the images from the paper

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# Figure: illustration of the domain shift for the different datasets with the proposed distance function
log_dir = 'ae'
x_ref = np.load(os.path.join(log_dir, 'x_ref.npy'))
z_ref = np.load(os.path.join(log_dir, 'z_ref.npy'))
z = np.load(os.path.join(log_dir, 'z.npy'))
doms = np.load(os.path.join(log_dir, 'doms.npy'))
samples = np.load(os.path.join(log_dir, 'samples.npy'))
reconstructions = np.load(os.path.join(log_dir, 'reconstructions.npy'))
closest_samples = np.load(os.path.join(log_dir, 'closest_samples.npy'))
closest_dists = np.load(os.path.join(log_dir, 'closest_dists.npy'))

# make figure for a reference domain
d = 0
k = closest_samples.shape[1]
n_domains = z.shape[0]
plt.imshow(x_ref[0], cmap='gray')
plt.title('Reference sample (domain %d)' % d)
plt.axis('off')
plt.show()
i = 1
for dd in range(n_domains):
    for kk in range(k):
        plt.subplot(n_domains, k, i)
        i += 1
        plt.imshow(closest_samples[dd, kk, ...], cmap='gray')
        # plt.title('%.2f' % closest_dists[dd, kk])
        plt.axis('off')
plt.show()

# constants
# domain names
EPFL = 'epfl'
EPFL_NAME = 'EPFL'
UROCELL = 'urocell'
UROCELL_NAME = 'UroCell'
PO936Q = 'po936q'
PO936Q_NAME = 'Po936Q'
MIRA = 'mira'
MIRA_NAME = 'MiRA'
MITOEM_H = 'mitoem-h'
MITOEM_H_NAME = 'MitoEM-H'
MITOEM_R = 'mitoem-r'
MITOEM_R_NAME = 'MitoEM-R'
KASTHURI = 'kasthuri'
KASTHURI_NAME = 'Kasthuri'
VNC = 'vnc'
VNC_NAME = 'VNC'
EMBL_HELA = 'embl-hela'
EMBL_HELA_NAME = 'EMBL-HeLa'
VIB_EVHELA = 'vib-evhela'
VIB_EVHELA_NAME = 'VIB-evHeLa'
VIB_SHPERK = 'vib-shperk'
VIB_SHPERK_NAME = 'VIB-shPerk'
# class labels
MITO = 'M'
ER = 'ER'
NM = 'NM'
# class colors
MITO_COLOR = 'tab:red'
ER_COLOR = 'tab:green'
NM_COLOR = 'tab:blue'


def load_results(basepath, domains, n_exp, metrics, classes):
    df_final = pd.DataFrame(columns=['measurement', 'metric', 'domain', 'experiment', 'class'])
    df_best = pd.DataFrame(columns=['measurement', 'metric', 'domain', 'experiment', 'class'])
    for i, dom in enumerate(domains):
        for n in range(n_exp):
            r_final = np.load(os.path.join(basepath, dom, str(n), 'validation_final.npy'))
            r_best = np.load(os.path.join(basepath, dom, str(n), 'validation_best.npy'))
            for j, metric in enumerate(metrics):
                df_final = df_final.append(
                    {'measurement': r_final.mean(axis=0)[j], 'metric': metric, 'domain': dom, 'experiment': n,
                     'class': classes[i]}, ignore_index=True)
                df_best = df_best.append(
                    {'measurement': r_best.mean(axis=0)[j], 'metric': metric, 'domain': dom, 'experiment': n,
                     'class': classes[i]}, ignore_index=True)
    return df_final, df_best


# Figure 1. Full-TD performance
full_td_img = '../train/exp/full-target-supervision/figure-1.pdf'
# data
domains_mito = [EPFL, UROCELL, PO936Q, MIRA, MITOEM_H, MITOEM_R, KASTHURI, VNC]
domain_names_mito = [EPFL_NAME, UROCELL_NAME, PO936Q_NAME, MIRA_NAME, MITOEM_H_NAME, MITOEM_R_NAME, KASTHURI_NAME, VNC_NAME]
domains_er = [EMBL_HELA, VIB_EVHELA + '-er', VIB_SHPERK + '-er']
domain_names_er = [EMBL_HELA_NAME, VIB_EVHELA_NAME, VIB_SHPERK_NAME]
domains_nm = [VIB_EVHELA + '-nm', VIB_SHPERK + '-nm']
domain_names_nm = [VIB_EVHELA_NAME, VIB_SHPERK_NAME]
domains = domains_mito + domains_er + domains_nm
domain_names = domain_names_mito + domain_names_er + domain_names_nm
classes = [MITO] * len(domains_mito) + [ER] * len(domains_er) + [NM] * len(domains_nm)
n_exp = 5
metrics = ['mIoU', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'F1 score']
basepath = '../train/exp/full-target-supervision'
_, df_best_mito = load_results(basepath, domains, n_exp, metrics, classes)

# plot everything
colors = len(domains_mito) * [MITO_COLOR] + len(domains_er) * [ER_COLOR] + len(domains_nm) * [NM_COLOR]
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
plt.yticks(locs, domain_names)
plt.ylabel(None)
plt.ylim([-0.5, len(domains) - 0.5])
plt.tight_layout()
# save file
plt.savefig(full_td_img, format='pdf')
plt.show()


# # Figure 2. Unsupervised domain adaptation: No-DA performance
# full_td_img = '../train/exp/unsupervised-da/no-da/figure-2.pdf'
# # data
# domains_mito = [EPFL, UROCELL, PO936Q, MIRA, KASTHURI, VNC]
# domain_names_mito = [EPFL_NAME, UROCELL_NAME, PO936Q_NAME, MIRA_NAME, KASTHURI_NAME, VNC_NAME]
# n_exp = 5
# metrics = ['mIoU', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'F1 score']
# basepath = '../train/exp/unsupervised-da/no-da'
# domains = []
# for src in domains_mito:
#     for tar in domains_mito:
#         if src != tar:
#             domains.append(src + '2' + tar)
# classes = [MITO] * len(domains)
# _, df_best_mito = load_results(basepath, domains, n_exp, metrics, classes)
#
# df = df_best_mito[df_best_mito['metric'] == 'mIoU'].groupby('domain').mean().reset_index()
# measurements = np.zeros((len(domains_mito), len(domains_mito))) + 0.5
# for i, src in enumerate(domains_mito):
#     for j, tar in enumerate(domains_mito):
#         if src != tar:
#             measurements[i, j] = df[df['domain'] == (src + '2' + tar)]['measurement'].values[0]
# plt.imshow(measurements)
# plt.show()
#
# # plot everything
# colors = len(domains_mito) * [MITO_COLOR]
# sns.set_style('whitegrid')
# ax = sns.barplot(data=df_best_mito[df_best_mito['metric'] == 'mIoU'], y='domain', x='measurement', palette=colors)
# # make it look nicer
# plt.xlabel('mIoU')
# # plt.xlim([0.6, 1.0])
# locs, labels = plt.yticks()
# plt.yticks(locs, domains)
# plt.ylabel(None)
# plt.ylim([-0.5, len(domains) - 0.5])
# plt.tight_layout()
# # save file
# # plt.savefig(full_td_img, format='pdf')
# plt.show()
