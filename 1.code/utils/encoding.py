import pandas as pd, sys, pickle, getpass
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder

username = getpass.getuser()
if(username == 'sakshigupta43'):
    sys.path.insert(0, '/Users/sakshigupta43/Desktop/Guild_Competition/ML_Guild_2020')
else:
        sys.path.insert(0, '/Users/skoranne/Desktop/Guild_Competition/ML_Guild_2020')

from CONFIG import *

'''Encoder I/O'''
def write_encoder(encoder, encoder_type = 'target',variable_name = 'default'):
    '''Takes the encoder and saves it as .pkl file'''
    with open(path_data_processed +'/encoders/' + encoder_type + '/' + variable_name + '.pkl', 'wb') as outfile:
        pickle.dump(encoder,outfile)
    return

def read_encoder(encoder_type = 'target', variable_name = 'default'):
    '''loads classes from an encoder'''
    return pickle.load(open(path_data_processed +'/encoders/' + encoder_type + '/' + variable_name + '.pkl','rb'))
      
'''Target Encoding'''
def target_encoding_fit(X,y,cols):
    '''Target/Mean Encoding - Takes X_train, y_train, columns to be encoded, saves encoded files '''
    for col in cols:
        print("Encoding for column: {}".format(col))
        encoder = TargetEncoder(cols = [col])
        encoder.fit(X[col], y)
        write_encoder(encoder, 'target', col)
    return

def target_encoding_transform(df, cols):
    for col in cols:
        print("Encoding for column: {}".format(col))
        try:
            encoder = read_encoder('target', col)
        except:
            print('Error occurred when reading column {}'.format(col))
        df[col] = encoder.transform(df[col])
    return df

'''OHE Encoding'''
def ohe_encoding_fit(X,y,cols):
    '''OHE - Takes X_train, y_train, columns to be encoded, saves encoded files '''
    for col in cols:
        print("Encoding for column: {}".format(col))
        encoder = TargetEncoder(cols = [col])
        encoder.fit(X[col], y)
        write_encoder(encoder, 'ohe', col)
    return

def ohe_encoding_transform(df, cols):
    for col in cols:
        print("Encoding for column: {}".format(col))
        try:
            encoder = read_encoder('ohe', col)
        except:
            print('Error occurred when reading column {}'.format(col))
        df[col] = encoder.transform(df[col])
    return df

'''Label Encoding'''
def label_encoding_fit(X,y,cols):
    '''Label - Takes X_train, y_train, columns to be encoded, saves encoded files '''
    for col in cols:
        print("Encoding for column: {}".format(col))
        encoder = TargetEncoder(cols = [col])
        encoder.fit(X[col], y)
        write_encoder(encoder, 'label', col)
    return

def label_encoding_transform(df, cols):
    for col in cols:
        print("Encoding for column: {}".format(col))
        try:
            encoder = read_encoder('label', col)
        except:
            print('Error occurred when reading column {}'.format(col))
        df[col] = encoder.transform(df[col])
    return df

'''CatBoost Encoding'''
def catboost_encoding_fit(X,y,cols):
    '''CatBoost - Takes X_train, y_train, columns to be encoded, saves encoded files '''
    for col in cols:
        print("Encoding for column: {}".format(col))
        encoder = CatBoostEncoder(cols = [col])
        encoder.fit(X[col], y)
        write_encoder(encoder, 'catboost', col)
    return

def catboost_encoding_transform(df, cols):
    for col in cols:
        print("Encoding for column: {}".format(col))
        try:
            encoder = read_encoder('catboost', col)
        except:
            print('Error occurred when reading column {}'.format(col))
        df[col] = encoder.transform(df[col])
    return df

'''James-Stein Encoding'''
def james_stein_encoding_fit(X,y,cols):
    '''James-Stein - Takes X_train, y_train, columns to be encoded, saves encoded files '''
    for col in cols:
        print("Encoding for column: {}".format(col))
        encoder = JamesSteinEncoder(cols = [col])
        encoder.fit(X[col], y)
        write_encoder(encoder, 'james_stein', col)
    return

def james_stein_encoding_transform(df, cols):
    for col in cols:
        print("Encoding for column: {}".format(col))
        try:
            encoder = read_encoder('james_stein', col)
        except:
            print('Error occurred when reading column {}'.format(col))
        df[col] = encoder.transform(df[col])
    return df



