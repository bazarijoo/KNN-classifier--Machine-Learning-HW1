import pandas as pd
import numpy as np

def loadDataset():
    data = pd.read_csv("breast-cancer-wisconsin.data", header=None,
                       names=['id', 'Clump Thickness', 'Uniformity of Cell Size',
                              'Uniformity of Cell Shape', 'Marginal Adhesion',
                              'Single Epithelial Cell Size', 'Bare Nuclei',
                              'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'], na_values='?')
    # handling missing values
    data['Bare Nuclei'] = data['Bare Nuclei'].fillna(value=data['Bare Nuclei'].median())
    data['Bare Nuclei'] = data['Bare Nuclei'].astype(np.int64)
    return data
