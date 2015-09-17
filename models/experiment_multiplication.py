import json
from data.data import Data
from utils import compare_models
import runner
from models.eloTreeDecay import EloTreeDecayModel
from skills import load_skills, load_questions, get_question_parents, get_skill_parents, split_multiplication_skills
import pylab as plt
import numpy as np



data = Data("data/matmat-all-2015-09-16.pd", train=0)
skills, questions = split_multiplication_skills()
qp, sp = get_question_parents(load_questions()), get_skill_parents(load_skills())


def compute_difficulties(model):
    runner.Runner(data, model).run()
    json.dump(model.difficulty, open("cache/difficulties.json", "w"))
    return model.difficulty

# compute_difficulties(EloTreeDecayModel(qp, sp, alpha=0.25, beta=0.02))


def get_avg_over_questions(a, b, data):
    skill_id = skills[skills["name"] == u"{}x{}".format(a, b)].index[0]
    sum, count = 0, 0
    for qid in questions[questions["skill"] == skill_id].index:
        sum += difficulties[str(qid)]
        count += 1
    return sum / count

def get_error_rate(a, b, data):
    skill_id = skills[skills["name"] == u"{}x{}".format(a, b)].index[0]
    return 1 - data[skill_id]

def plot_table(data):
    plt.figure()
    plt.pcolor(data)
    plt.xticks(np.arange(1, 11) - 0.5, range(1, 11))
    plt.yticks(np.arange(1, 11) - 0.5, range(1, 11))
    plt.ylabel("first")
    plt.xlabel("second")
    plt.colorbar()

difficulties = json.load(open("cache/difficulties.json"))
success_rates = data.get_dataframe_all().join(questions, on="item").groupby("skill")["correct"].mean()
# diffi = np.array([[get_avg_over_questions(a, b, difficulties) for b in range(1, 11)] for a in range(1, 11)])
# diffe = np.array([[get_avg_over_questions(a, b,difficulties) - get_avg_over_questions(b, a, difficulties) for b in range(1, 11)] for a in range(1, 11)])
# diffi = np.array([[get_error_rate(a, b, success_rates) for b in range(1, 11)] for a in range(1, 11)])
diffe = np.array([[get_error_rate(a, b, success_rates) - get_error_rate(b, a, success_rates) for b in range(1, 11)] for a in range(1, 11)])

plot_table(diffe)

plt.show()