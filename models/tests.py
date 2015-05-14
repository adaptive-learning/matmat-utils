from data.data import Data
from evaluator import Evaluator
from models.model import AvgModel, AvgItemModel
import pylab as plt
import seaborn as sbn
from models.elo import EloModel
from models.eloTree import EloTreeModel
from models.eloCurrent import EloPriorCurrentModel
from models.eloTime import EloTimeModel
from runner import Runner
from skills import load_skills, get_skill_parents, get_question_parents, load_questions
from utils import elo_grid_search, compare_models, compare_brier_curve, get_skills
import pandas as pd


data = Data("data/matmat-all.pd", train=0)
qp, sp = get_question_parents(load_questions()), get_skill_parents(load_skills())
# model = EloTreeModel(get_question_parents(load_questions()), skill_parents=get_skill_parents(load_skills()))
# model = EloPriorCurrentModel()

def fit_time_decay():
    slopes = [0.6, 0.8, 0.85, 0.9, 0.95, 1., 1.05, 1.1, 1.2]
    rmses = []

    for slope in slopes:
        model = EloTimeModel(alpha=0.8, beta=0, time_penalty_slope=slope)

        Runner(data, model).run()
        rmses.append(Evaluator(data, model).evaluate()["rmse"])

    plt.plot(slopes, rmses)


def skill_comparision():
    model = EloTreeModel(qp, sp, alpha=1.2, beta=0.1, KC=3.5, KI=2.5)
    Runner(data, model).run()
    skills = model.skill
    skill1 = pd.Series(skills[26], name="plus")
    skill2 = pd.Series(skills[209], name="times")
    common = list(set(skill1.index) & set(skill2.index))
    plt.plot(skill1[common], skill2[common], ".")
    plt.xlabel(skill1.name)
    plt.ylabel(skill2.name)

plt.show()