import numpy as np
import pylab as plt
import pandas as pd
from models.elo import EloModel
from evaluator import Evaluator
from runner import Runner


def elo_grid_search(data, alpha_range=(0.4, 2, 0.2), beta_range=(0.02, 0.2, 0.02), run=False):
    alphas = np.arange(*alpha_range)
    betas = np.arange(*beta_range)

    results = pd.DataFrame(columns=alphas, index=betas, dtype=float)
    plt.figure()
    for alpha in alphas:
        for beta in betas:

            model = EloModel(alpha=alpha, beta=beta)
            # model = EloTreeModel(alpha=alpha, beta=beta, clusters=utils.get_maps("data/"), local_update_boost=0.5)
            if run:
                Runner(data, model).run()
                report = Evaluator(data, model).evaluate()
            else:
                report = Evaluator(data, model).get_report()
            # results[alpha][beta] = report["brier"]["reliability"]
            results[alpha][beta] = report["rmse"]
    plt.title(data)

    cmap = plt.cm.get_cmap("gray")
    cmap.set_gamma(0.5)
    plt.pcolor(results, cmap=cmap)
    plt.yticks(np.arange(0.5, len(results.index), 1), results.index)
    plt.ylabel("betas")
    plt.xticks(np.arange(0.5, len(results.columns), 1), results.columns)
    plt.xlabel("alphas")
    plt.colorbar()