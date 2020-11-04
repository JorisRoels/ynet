# this scripts generates the tables from the paper

import pandas as pd
import numpy as np
import os

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