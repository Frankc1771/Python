import pandas as pd
import os
import numpy as np
import SetDiagCodes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#get to the right directory
#os.chdir('DAT203.2\Module 1')
#read csv dataframe
print(os.getcwd())
df = pd.read_csv('cleanedDiabetesData.csv')


# add visualization
def diabetes_bar(df):   
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    
## Create a series of bar plots for the various levels of the
## string columns in the data frame by readmi_class. 
    names = df.columns.tolist()
    for col in names:
        if(df[col].dtype not in [np.int64, np.int32, np.float64]):
            temp1 = df.loc[df.readmitted == 'YES', col].value_counts()
            temp0 = df.loc[df.readmitted == 'NO', col].value_counts() 
        
            fig = plt.figure(figsize = (12,6))
            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax0 = fig.add_subplot(1, 2, 2) 
            temp1.plot(kind = 'bar', ax = ax1)
            ax1.set_title('Values of ' + col + '\n for readmitted patients')
            temp0.plot(kind = 'bar', ax = ax0)
            ax0.set_title('Values of ' + col + '\n for patients not readmitted')
            fig.savefig('bar_' + col + '.png')     
    return 'Done'


def diabetes_box(df):
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import numpy as np
    import matplotlib.pyplot as plt
    
## Now make some box plots of the columbns with numerical values.
    names = df.columns.tolist()
    for col in names:
        if(df[col].dtype in [np.int64, np.int32, np.float64]):
            temp1 = df.loc[df.readmitted == 'YES', col]
            temp0 = df.loc[df.readmitted == 'NO', col]  
             
            fig = plt.figure(figsize = (12,6))
            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax0 = fig.add_subplot(1, 2, 2) 
            ax1.boxplot(temp1.as_matrix())
            ax1.set_title('Box plot of ' + col + '\n for readmitted patients')
            ax0.boxplot(temp0.as_matrix())
            ax0.set_title('Box plot of ' + col + '\n for patients not readmitted')
            fig.savefig('box_' + col + '.png')

    return 'Done'
    
def diabetes_hist(df):
    import matplotlib
    matplotlib.use('agg')  # Set backend
    import numpy as np
    import matplotlib.pyplot as plt
    
## Now make historgrams of the columbns with numerical values.
    names = df.columns.tolist()
    for col in names:
        if(df[col].dtype in [np.int64, np.int32, np.float64]):
            temp1 = df.loc[df.readmitted == 'YES', col]
            temp0 = df.loc[df.readmitted == 'NO', col]  
             
            fig = plt.figure(figsize = (12,6))
            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax0 = fig.add_subplot(1, 2, 2) 
            ax1.hist(temp1.as_matrix(), bins = 30)
            ax1.set_title('Histogram of ' + col + '\n for readmitted patients')
            ax0.hist(temp0.as_matrix(), bins = 30)
            ax0.set_title('Histogram of ' + col + '\n for patients not readmitted')
            fig.savefig('hist_' + col + '.png')

    return 'Done'    


#diabetes_bar(df)


















#change categorical data to numeric to run regression model


df = pd.get_dummies(df, columns = ['race', 'gender', 'age', 'weight', 'payer_code', 'medical_specialty',
                                   'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide',
                                   'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
                                   'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
                                   'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone',
                                   'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed', 'admission_type_description'])

df.readmitted =  df.readmitted.map({'NO':0, 'YES':1})


# seperate label column
y = df['readmitted']
X = df.drop(['readmitted'], axis = 1)
# split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4)
#two way regression model and score model. Add a confusion matrix
#        and check different statistics
model = LogisticRegression(C = .001, random_state = 1234, tol =1e-07)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('score',score)
y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
accuracy = (tp+tn)/(tp+tn+fp+fn)
print('matrix', confusion_matrix(y_test, y_pred))



