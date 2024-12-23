import pandas as pd

def get_highest_correlation(df, target):
    correlations = df.corr()[target].drop(target)
    return correlations.idxmax()