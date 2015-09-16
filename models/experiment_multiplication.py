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

difficulties = json.load(open("cache/difficulties.json"))

def get_avg_difficulty(a, b):
    skill_id = skills[skills["name"] == u"{}x{}".format(a, b)].index[0]
    sum, count = 0, 0
    for qid in questions[questions["skill"] == skill_id].index:
        sum += difficulties[str(qid)]
        count += 1
    return sum / count

if 1:
    d = np.array([[get_avg_difficulty(a, b) for b in range(1, 11)] for a in range(1, 11)])
    plt.pcolor(d)
    plt.xticks(np.arange(1, 11) - 0.5, range(1, 11))
    plt.yticks(np.arange(1, 11) - 0.5, range(1, 11))
    plt.ylabel("first")
    plt.xlabel("second")
    plt.colorbar()

if 1:
    plt.figure()
    d = np.array([[get_avg_difficulty(a, b) - get_avg_difficulty(b, a) for b in range(1, 11)] for a in range(1, 11)])
    plt.pcolor(d)
    plt.xticks(np.arange(1, 11) - 0.5, range(1, 11))
    plt.yticks(np.arange(1, 11) - 0.5, range(1, 11))
    plt.ylabel("first")
    plt.xlabel("second")
    plt.colorbar()

plt.show()