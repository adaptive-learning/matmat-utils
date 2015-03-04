import pandas as pd


def load_skills(csv_file="data/model_skill.csv"):
    """
    Loads skills form csv to pandas DataFrame

    Args:
        csv_file (str): path to source file
    Returns:
        pandas.DataFrame
    """
    df = pd.DataFrame.from_csv(csv_file)

    return df