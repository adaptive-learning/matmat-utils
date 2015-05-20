import numpy as np
import pylab as plt
import pandas as pd
from models.elo import EloModel
from evaluator import Evaluator
from models.eloCurrent import EloPriorCurrentModel
from models.eloTree import EloTreeModel
from models.eloTreeDecay import EloTreeDecayModel
from runner import Runner
from skills import get_question_parents, load_questions, get_skill_parents, load_skills


def elo_grid_search(data, alpha_range=(0.4, 2, 0.2), beta_range=(0.02, 0.2, 0.02), model_class=EloModel):
    alphas = np.arange(*alpha_range)
    betas = np.arange(*beta_range)

    results = pd.DataFrame(columns=alphas, index=betas, dtype=float)
    plt.figure()
    for alpha in alphas:
        for beta in betas:
            if model_class == EloModel:
                model = EloModel(alpha=alpha, beta=beta)
            if model_class == EloPriorCurrentModel:
                model = EloPriorCurrentModel(alpha=alpha, beta=beta)
            if model_class == EloTreeModel:
                model = EloTreeModel(get_question_parents(load_questions()), get_skill_parents(load_skills()), alpha=alpha, beta=beta)
            if model_class == EloTreeDecayModel:
                model = EloTreeDecayModel(get_question_parents(load_questions()), get_skill_parents(load_skills()), alpha=alpha, beta=beta)
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


def elo_pfa_search(data, kc=(0, 5, 0.5), ki=(0, 5, 0.5), model_class=EloPriorCurrentModel):
    kcs = np.arange(*kc)
    kis = np.arange(*ki)

    results = pd.DataFrame(columns=kcs, index=kis, dtype=float)
    plt.figure()
    for kc in kcs:
        for ki in kis:
            if model_class == EloPriorCurrentModel:
                model = EloPriorCurrentModel(KC=kc, KI=ki)
            if model_class == EloTreeModel:
                model = EloTreeModel(get_question_parents(load_questions()), get_skill_parents(load_skills()), KC=kc, KI=ki)
            report = Evaluator(data, model).get_report()
            # results[kc][ki] = report["brier"]["reliability"]
            results[kc][ki] = report["rmse"]
    plt.title(data)

    cmap = plt.cm.get_cmap("gray")
    cmap.set_gamma(0.5)
    plt.pcolor(results, cmap=cmap)
    plt.yticks(np.arange(0.5, len(results.index), 1), results.index)
    plt.ylabel("K for incorrect")
    plt.xticks(np.arange(0.5, len(results.columns), 1), results.columns)
    plt.xlabel("K for correct")
    plt.colorbar()

def compare_models(data, models, dont=False, resolution=True, auc=False, evaluate=False, diff_to=None):
    if dont:
        return
    plt.xlabel("RMSE")
    if auc:
        plt.ylabel("AUC")
    elif resolution:
        plt.ylabel("Resolution")
    else:
        plt.ylabel("Brier score")
    for model in models:
        if evaluate:
            Evaluator(data, model).evaluate()
        report = Evaluator(data, model).get_report()
        print model
        print "RMSE: {:.5}".format(report["rmse"])
        if diff_to is not None:
            print "RMSE diff: {:.5f}".format(diff_to - report["rmse"])
        print "LL: {:.6}".format(report["log-likely-hood"])
        print "AUC: {:.4}".format(report["AUC"])
        print "Brier resolution: {:.4}".format(report["brier"]["resolution"])
        print "Brier reliability: {:.3}".format(report["brier"]["reliability"])
        print "Brier uncertainty: {:.3}".format(report["brier"]["uncertainty"])
        print "=" * 50

        x = report["rmse"]
        if auc:
            y = report["AUC"]
        elif resolution:
            y = report["brier"]["resolution"]
        else:
            y = report["brier"]["reliability"] - report["brier"]["resolution"] + report["brier"]["uncertainty"]
        plt.plot(x, y, "bo")
        plt.text(x, y, model, rotation=0, )


def compare_brier_curve(data, model1, model2):
    # Evaluator(data, model2).evaluate()
    report1 = Evaluator(data, model1).get_report()
    report2 = Evaluator(data, model2).get_report()

    fig, ax1 = plt.subplots()
    ax1.plot([]+report1["zextra"]["brier"]["bin_prediction_means"], []+report1["zextra"]["brier"]["bin_correct_means"], "g", label="M1")
    ax1.plot([]+report2["zextra"]["brier"]["bin_prediction_means"], []+report2["zextra"]["brier"]["bin_correct_means"], "r", label="M2")
    ax1.plot((0, 1), (0, 1), "k--")

    ax2 = ax1.twinx()

    bin_count = report1["zextra"]["brier"]["bin_count"]
    counts1 = np.array(report1["zextra"]["brier"]["bin_counts"])
    counts2 = np.array(report2["zextra"]["brier"]["bin_counts"])
    bins = (np.arange(bin_count) + 0.5) / bin_count
    ax2.bar(bins, counts1, width=(0.45 / bin_count), alpha=0.2, color="g")
    ax2.bar(bins-0.023, counts2, width=(0.45 / bin_count), alpha=0.2, color="r")
    # plt.bar(bins, (counts1 - counts2) / max(max(counts2),max(counts1)), width=(0.5 / bin_count), alpha=0.8, color="r")

    plt.title("M1: {}\nM2: {}".format(model1, model2))
    plt.xticks(list(bins-0.025) + [1.])

    ax1.legend(loc=2)


def get_skills(data, model):
    Runner(data, model).run()
    if model.__class__ == EloModel:
        return model.global_skill
    return None