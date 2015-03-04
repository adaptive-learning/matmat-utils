import pandas as pd


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
    Remove answers with long and negative solving time

    Args:
        answers (pandas.DataFrame)
        max_time: maximal time in seconds
    Returns:
        pandas.DataFrame
    """

    return answers[(0 <= answers["solving_time"]) & (answers["solving_time"] < max_time)]
