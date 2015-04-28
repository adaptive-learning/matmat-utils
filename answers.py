import pandas as pd
import numpy as np


def load_answers(csv_file="data/questions_answer.csv"):
    """
    Loads answers form csv to pandas DataFrame

    Args:
        csv_file (str): path to source file
    Returns:
        pandas.DataFrame
    """

    df = pd.DataFrame.from_csv(csv_file)

    return df


def filter_long_times(answers, max_time=100):
    """
    Remove answers with long and non-positive solving time

    Args:
        answers (pandas.DataFrame)
        max_time: maximal time in seconds
    Returns:
        pandas.DataFrame
    """

    return answers[(0 < answers["solving_time"]) & (answers["solving_time"] < max_time)]


def filter_users(answers, min_answers_per_user=10, min_answers_per_item=1):
    answers = answers[answers.join(pd.Series(answers.groupby("question").apply(len), name="count"), on="question")["count"] > min_answers_per_item]
    answers = answers[answers.join(pd.Series(answers.groupby("user").apply(len), name="count"), on="user")["count"] > min_answers_per_user]
    return answers


def join_response_time_means(answers, of="user"):
    answers["log_times"] = np.log(answers["solving_time"])
    return answers.join(np.exp(pd.Series(answers.groupby(of)["log_times"].apply(np.mean), name="{}_rt_mean".format(of))), on=of)


def join_success_rates(answers, of="user"):
    return answers.join(pd.Series(answers.groupby(of)["correctly_solved"].apply(np.mean), name="{}_success_rate".format(of)), on=of)


def join_answers_count(answers, of="user"):
    return answers.join(pd.Series(answers.groupby(of).apply(len), name="{}_answer_count".format(of)), on=of)


def convert_answers_to_model(answers):
    answers = filter_long_times(filter_users(answers))
    answers["correct"] = answers["correctly_solved"]
    answers["student"] = answers["user"]
    answers["item"] = answers["question"]
    answers["response_time"] = answers["solving_time"]
    answers["id"] = answers.index
    del answers["correctly_solved"], answers["user"], answers["question"], answers["solving_time"]

    return answers


def get_geography_answers(filename="../thrans/model comparison/data/raw data/geography-all.csv", min_answers_per_item=1000, min_answers_per_user=10):
    answers = pd.DataFrame.from_csv(filename, index_col=False)
    answers["solving_time"] = answers["response_time"] / 1000.
    answers["log_times"] = np.log(answers["solving_time"])
    answers["correctly_solved"] = answers["place_asked"] == answers["place_answered"]
    answers["question"] = answers["place_asked"]
    answers = answers[answers.join(pd.Series(answers.groupby("question").apply(len), name="count"), on="question")["count"] > min_answers_per_item]
    answers = answers[answers.join(pd.Series(answers.groupby("user").apply(len), name="count"), on="user")["count"] > min_answers_per_user]

    return answers