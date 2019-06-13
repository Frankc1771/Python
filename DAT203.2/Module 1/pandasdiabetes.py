import pandas as pd
import os
import numpy as np
import SetDiagCodes
from sklearn.preprocessing import StandardScaler
#get to the right directory
#os.chdir('DAT203.2\Module 1')
#read csv dataframe
print(os.getcwd())
df = pd.read_csv('diabetic_data.csv')
admissions = pd.read_csv('admissions_mapping.csv')
'''
df['admission_type_id'] = df.admission_type_id.map({#add options here})
'''
   

#create a dictionary for admissions id mapped to description
x = admissions['admission_type_id'].tolist()
y = admissions['admission_type_description'].tolist()
keyDic = {}
for i,j in enumerate(y):
    keyDic[i+1] = j
for i in keyDic.keys():
    if keyDic[i] == 'Not Available' or keyDic[i] == 'Not Mapped' or keyDic[i] == 'NULL':
        keyDic[i] = 'unknown'
    if pd.isnull(keyDic[i]):
        keyDic[i] = 'unknown'
     
#join the description to the id column of the diabetes df

for i,j in enumerate(df['admission_type_id']):
    df.loc[i, 'admission_type_description'] = keyDic[j]

#replace any missing strings with unknown and any missing numeric values with 0 
for col in df.columns:
    if df[col].dtype in [np.int64, np.int32, np.float64]:
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna('unknown')
    
#fix diagnosis codes and change label column readmitted to just yes or no
codes = SetDiagCodes.create_map()
col_list = ['diag_1', 'diag_2', 'diag_3']
df[col_list] = SetDiagCodes.map_codes(df[col_list], codes)

for i in range(len(df['readmitted'])):
    if df['readmitted'][i] == 'NO':
        df.loc[i, 'readmitted'] = 'NO'
    else:
        df.loc[i, 'readmitted'] = 'YES'
        
#drop some useless columns
df = df.drop(['encounter_id', 'patient_nbr', 'admission_type_id'], axis = 1)


#normalize numeric columns
#TRY RUNNING THIS TO SEE IF IT WORKS? OTHERWISE WILL HAVE TO LOOP THROUGH
#NUMERIC COLUMNS AND APPLY IT OR MANUAL WRITE CODE TO APPLY ZSCORE TRANSFORMATION WITHOUT SKLEARN
zscore = StandardScaler()
#convert object classes to dummy classes

for col in df.columns:
    if df[col].dtype in [np.int64, np.int32, np.float64]:
        zscore.fit(df[col].values.reshape(-1,1))
        df[col] = zscore.transform(df[col].values.reshape(-1,1))
    else:
        df[col] = df[col].astype('category')

export_csv = df.to_csv('cleanedDiabetesData.csv', index = False) 
