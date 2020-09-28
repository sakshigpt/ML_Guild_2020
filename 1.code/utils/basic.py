import pandas as pd, numpy as np
import datetime
import scipy.stats as ss


def convert_to_date(df, col):
    df[col] = pd.to_datetime(df[col])
    return df

def get_date_feats(df, col):
    df[col] = convert_to_date(df, col)
    df[str(col) + '_month'] = df[col].dt.month
    df[str(col) + '_day_of_week'] = df[col].dt.dayofweek
    df[str(col) + '_date'] = df[col].dt.date
    df.drop(columns = col, inplace = True)
    return df

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))