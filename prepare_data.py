from collections import defaultdict
import json
import pandas as pd
import os


def get_skill_parents(skills):
    map = {}
    for id, skill in skills.iterrows():
        map[id] = int(skill["parent"]) if not pd.isnull(skill["parent"]) else None
    return map

def get_skill_parent_lists(skills):
    map = get_skill_parents(skills)
    lists = defaultdict(lambda: [])
    for skill in map:
        s = skill
        while s:
            lists[skill].append(s)
            s = map[s]
    return lists


def prepare_data(input_dir="data/source", output_dir="data"):
    answers = pd.DataFrame.from_csv(os.path.join(input_dir, "questions_answer.csv"))
    items = pd.DataFrame.from_csv(os.path.join(input_dir, "questions_question.csv"))
    simulators = pd.DataFrame.from_csv(os.path.join(input_dir, "questions_simulator.csv"))
    skills = pd.DataFrame.from_csv(os.path.join(input_dir, "model_skill.csv"))
    skill_parents = get_skill_parent_lists(skills)

    items = items.join(simulators, "player")
    items.rename(inplace=True, columns={"name": "visualization"})
    items["answer"] = 0
    items["skill_lvl_1"], items["skill_lvl_2"], items["skill_lvl_3"] = None, None, None
    for id, item in items.iterrows():
        data = json.loads(item["data"])
        items.loc[id, "data"] = item["data"].replace('"', "'")
        items.loc[id, "answer"] = int(data["answer"])
        for i, skill in enumerate(skill_parents[item["skill"]][::-1][1:]):
            items.loc[id, "skill_lvl_{}".format(i + 1)] = skill
    items = items[["answer", "visualization", "skill", "skill_lvl_1", "skill_lvl_2", "skill_lvl_3", "data"]]

    answers.rename(inplace=True, columns={
        "question": "item",
        "user": "student",
        "solving_time": "response_time",
        "correctly_solved": "correct",
    })
    answers = answers.join(items[["answer"]], on="item", rsuffix="_expected")
    answers = answers[["timestamp", "item", "student", "response_time", "correct", "answer", "answer_expected", "device", "log"]]
    answers = answers.round({"response_time": 3})

    skills.rename(inplace=True, columns={"note": "name_cz",})
    skills = skills[["name", "parent"]]

    simulators.rename(inplace=True, columns={"note": "name_cz",})

    answers.to_csv(os.path.join(output_dir, "answers.csv"))
    items.to_csv(os.path.join(output_dir, "items.csv"))
    # simulators.to_csv(os.path.join(output_dir, "visualizations.csv"))
    skills.to_csv(os.path.join(output_dir, "skills.csv"), float_format="%.0f")


prepare_data()