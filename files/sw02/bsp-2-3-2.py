# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 11:21:35 2019

@author: tascheib
"""

import pandas as pd  
import numpy as np
from sklearn.preprocessing import Imputer 
data = {'Name': ['John','Paul', np.NaN, 'Wale', 'Mary', 'Carli', 'Steve'], 
        'Age': [21,23,np.nan,19,25,np.nan,15],
        'Sex': ['M',np.nan,np.nan,'M','F','F','M'],
        'Goals': [5,10,np.nan,19,5,0,7],
        'Assists': [7,4,np.nan,9,7,6,4],
        'Value': [55,84,np.nan,90,63,15,46]}  
df=pd.DataFrame(data, columns =['Name','Age','Sex',
                                'Goals', 'Assists', 'Value'])
values = df[["Age","Goals","Assists","Value"]].values
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0, verbose = 0, copy = True)
transformed_values = imputer.fit_transform(values)
df_new = pd.DataFrame(transformed_values, columns =['Age','Goals', 'Assists', 'Value'])
print(df_new)
df_new
