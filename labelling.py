import pandas as pd
import numpy as np

path_TPM = 'pnas.1814551116.sd02.xlsx'

TPM = pd.read_excel(path_TPM)
TPM['label'] = ['unexpressed' if x < 1.0 else 'expressed'
                for x in TPM['max_TPM'] ]

TPM.to_csv('labels.csv')
