import argparse
import numpy as np
import os
from neuralnets.util.io import print_frm

parser = argparse.ArgumentParser()
parser.add_argument("--basepath", help="Base path of the domains", type=str,
                    default="../train/exp/full-target-supervision")
parser.add_argument("--domains", help="Domains", type=str, default="epfl,mira,vnc,embl-hela")
parser.add_argument("--n_exp", help="Number of experiments", type=int, default=5)
args = parser.parse_args()
domains = [item for item in args.domains.split(',')]

# parameters
basepath = args.basepath
n_exp = args.n_exp
metrics = ['mIoU', 'Accuracy', 'Balanced accuracy', 'Precision', 'Recall', 'F1 score']

# load results and save a dataframe
results_final = []
results_best = []
for i, dom in enumerate(domains):
    results_dom_final = []
    results_dom_best = []
    for n in range(n_exp):
        r_final = np.load(os.path.join(basepath, dom, str(n), 'validation_final.npy'))
        r_best = np.load(os.path.join(basepath, dom, str(n), 'validation_best.npy'))
        results_dom_final.append(r_final)
        results_dom_best.append(r_best)
    results_final.append(results_dom_final)
    results_best.append(results_dom_best)
results_final = np.asarray(results_final)  # D x N x C x M
results_best = np.asarray(results_best)  # D x N x C x M

# average over classes
results_final = 100 * np.mean(results_final, axis=2)
results_best = 100 * np.mean(results_best, axis=2)

# compute average and standard deviation over experiments
results_final_mean = np.mean(results_final, axis=1)
results_final_std = np.std(results_final, axis=1)
results_best_mean = np.mean(results_best, axis=1)
results_best_std = np.std(results_best, axis=1)

# report mean performance
for i, dom in enumerate(domains):
    print_frm('Domain: %s' % dom)
    print_frm('')

    print_frm('Validation performance final: ')
    for j, metric in enumerate(metrics):
        print_frm('    - %s: %.2f (+/- %.2f)' % (metric, results_final_mean[i, j], results_final_std[i, j]))
    print_frm('')

    print_frm('Validation performance best: ')
    for j, metric in enumerate(metrics):
        print_frm('    - %s: %.2f (+/- %.2f)' % (metric, results_best_mean[i, j], results_best_std[i, j]))
    print_frm('')

    print_frm('=================================================')
    print_frm('')
