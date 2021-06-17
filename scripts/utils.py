import pandas as pd


def load_dataset(path, cols):
    df = pd.read_csv(path, usecols=cols, names=cols)

    return df
