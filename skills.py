import os
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
