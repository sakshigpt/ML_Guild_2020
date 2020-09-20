import pandas as pd
import datetime

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