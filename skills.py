import json
import os
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)

def load_skills(csv_file=None):
    """
    Loads skills form csv to pandas DataFrame

    Args:
        csv_file (str): path to source file
    Returns:
        pandas.DataFrame
    """
    if csv_file is None:
        csv_file = os.path.join(BASE_DIR, "data/model_skill.csv")
    df = pd.DataFrame.from_csv(csv_file)

    return df


def load_questions(csv_file=None):
    if csv_file is None:
        csv_file = os.path.join(BASE_DIR, "data/questions_question.csv")
    df = pd.DataFrame.from_csv(csv_file)
    return df


def get_skill_parents(skills):
    map = {}
    for id, skill in skills.iterrows():
        map[id] = int(skill["parent"]) if not pd.isnull(skill["parent"]) else None
    return map


def get_question_parents(question):
    return dict(zip(question.index, question["skill"]))


def split_multiplication_skills():
    SMALL_MULTIPLICATION_ID = 210
    skills = load_skills()
    questions = load_questions()
    for id, skill in skills[skills["parent"] == SMALL_MULTIPLICATION_ID].iterrows():
        factors = map(int, re.search(r'(\d+)x(\d+)', skill["name"]).groups())
        if factors[0] == factors[1]:
            continue
        new = skill.copy()
        new["name"] = "{0[1]}x{0[0]}".format(factors)
        new["note"] = "{0[1]}&times;{0[0]}".format(factors)

        new_id = skills.index.max() + 1
        skills.loc[new_id] = new
        all, changed = 0, 0
        for qid, question in questions[questions["skill"] == id].iterrows():
            all += 1
            data = json.loads(question["data"])
            q = data[u"question"] if u"question" in data else data[u"text"]
            if (type(q) is list and q[2] == factors[0] and q[0] == factors[1]) \
                    or (type(q) is unicode and (q == u"{0[1]} &times; {0[0]}".format(factors)
                                                or q == u"{0[1]}&times;{0[0]}".format(factors))):
                questions.ix[qid, "skill"] = new_id
                changed += 1
        # if all - changed != changed:
        #     print all, changed

    return skills, questions
