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
# model = EloTreeModel(get_question_parents(load_questions()), skill_parents=get_skill_parents(load_skills()))
# model = EloPriorCurrentModel()

slopes = [0.95, 1.05, 0.6, 0.8, 0.9, 1, 1.1, 1.2]
rmses = []

for slope in slopes:
    model = EloTimeModel(alpha=0.8, beta=0, time_penalty_slope=slope)

    Runner(data, model).run()
    rmses.append(Evaluator(data, model).evaluate()["rmse"])

plt.plot(slopes, rmses)
plt.show()