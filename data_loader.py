import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from datasets import Dataset
import warnings
# Set pandas display options
pd.set_option("display.max_colwidth", 200)
# Ignore warnings
warnings.filterwarnings("ignore")

def read_data(file_path):
    data=pd.read_csv(file_path)

    data.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
    data.dropna(axis=0,inplace=True) #dropping na

    print(data.shape)

    print(data.info())
    
if __name__ == '__main__':
    read_data()