import numpy as np
import pandas as pd
from data_preprocesser import clean_data, create_training_data
import warnings
warnings.filterwarnings("ignore")

def read_data(file_path):
    data=pd.read_csv(file_path)

    data.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
    data.dropna(axis=0,inplace=True) #dropping na

    print(data.shape)

    print(data.info())
    
if __name__ == '__main__':
    read_data()