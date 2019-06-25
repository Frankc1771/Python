#imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

#load dataset
df = pd.read_csv('Bike Rental UCI dataset.csv')

#convert days from ints to strings
days = pd.DataFrame([[0, 1, 2, 3, 4, 5, 6],
                        ["Sun", "Mon", "Tue", "Wed", "Thr", "Fri", "Sat"]]).transpose()
days.columns = ['weekday', 'dayOfWeek'] 

#merge the day column to get accurate days of the week 
df = df.merge(days, left_on='weekday', right_on='weekday', how = 'left')

#create a new column that tracks days since the dataset started
df['days'] = pd.Series(range(df.shape[0]))/24
df['days'] = df['days'].astype('int')

#drop features that don't contribute
df = df.drop(['instant', 'dteday', 'weekday', 'atemp', 'casual', 'registered'], axis = 1)



#use zscore for numeric columns
num_cols = ['temp', 'hum', 'windspeed']
z_score = StandardScaler()
for col in num_cols:
    z_score.fit(df[col].values.reshape(-1,1))
    df[col] = z_score.transform(df[col].values.reshape(-1,1))

#check for NANS in each column
#df.isnull().sum()
#NO NANS in this dataframe

#seperate label column
y = df['cnt']
X = df.drop(['cnt', 'dayOfWeek'], axis = 1)
#split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

#linear regression model and score model before BONUS section was completed
'''
model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('r2', score)
y_pred = model.predict(X_test)
y_mean = []
for i in y_pred:
    y_mean.append(189.46)
y_mean = np.array(y_mean)

#appended column cnt mean to all predicted guesses to use for simple predictor 
#RELATIVE SQUARED ERROR and ROOT MEAN SQUARED ERROR
simple_pred = y_mean
squared_error = mean_squared_error(y_test, y_pred)
root_mean_squared = squared_error ** .5
simple_pred_error = mean_squared_error(y_test, y_mean)

#prediction model squared error / simple predictor squared error = relative square error
relative_squared_error = squared_error / simple_pred_error

print('root mean square', root_mean_squared)
print('relative squared error', relative_squared_error)

#mean absolute error
absolute_square_error = mean_absolute_error(y_test, y_pred)

print('mean absolute error', absolute_square_error)
#relative absolute error = absolute_square error / root_mean squared
relative_absolute_error = absolute_square_error / root_mean_squared
print('relative absolute error', relative_absolute_error)

#add predicted column to X_test
X_test['predicted'] = y_pred
'''










#visualizations
num_cols = ["temp", "hum", "windspeed", "hr"] 
           
def bike_scatter(df, cols):
    import matplotlib.pyplot as plt
    import statsmodels.nonparametric.smoothers_lowess as lw
    
    ## Loop over the columns and create the scatter plots
    for col in cols:
        ## first compute a lowess fit to the data
        los = lw.lowess(df['cnt'], df[col], frac = 0.3)
    
        ## Now make the plots
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df.plot(kind = 'scatter', x = col, y = 'cnt', ax = ax, alpha = 0.05)
        plt.plot(los[:, 0], los[:, 1], axes = ax, color = 'red')
        ax.set_xlabel(col)
        ax.set_ylabel('Number of bikes')
        ax.set_title('Number of bikes vs. ' + col)
    plt.show()
    return 'Done'        

cat_cols = ['season', 'yr', 'mnth', 'hr', 'holiday',
           'workingday', 'weathersit', 'dayOfWeek']
    
def bike_box(df, cols):
    import matplotlib.pyplot as plt
    
    ## Loop over the columns and create the box plots
    for col in cols:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df.boxplot(column = 'cnt', by = col, ax = ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Number of bikes')
        ax.set_title('Number of bikes vs. ' + col)
    plt.show()
    return 'Done'
 
plt_times = [6, 8, 10, 12, 14, 16, 18, 20]
def bike_series(df, tms):
    import matplotlib.pyplot as plt
        
    lims = (min(df.cnt), max(df.cnt))   
    for t in tms:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df[df.hr == t].plot(kind = 'line', x = 'days', y = 'cnt',
                            ylim = lims, ax = ax)    
        plt.xlabel("Days from start")
        plt.ylabel("Bikes rented")
        plt.title("Bikes rented by day for hour = " + str(t))
    plt.show()
    return 'Done'
  
hist_cols = ["cnt", "temp", "hum", "windspeed"] 
def bike_hist(df, cols):    
    import matplotlib.pyplot as plt
    
    ## Loop over columns and plot histograms
    for col in cols:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df[col].hist(bins = 30, ax = ax)
        ax.set_xlabel(col)
        ax.set_ylabel('Density of ' + col)
        ax.set_title('Density of ' + col)
    plt.show()
    
    return 'Done'
   
def bike_hist_cond(df, col, by):
    import matplotlib.pyplot as plt
    
    df = df[df.hr.isin(by)]
    ## Plot conditioned histograms
    fig = plt.figure(figsize=(10, 8))
    ax = fig.gca()
    df[[col, 'hr']].hist(bins = 30, by = ['hr'], ax = ax)
    plt.show()
    return 'Done'
     


#predictor visualizations

def ts_bikes(df, times):
    import matplotlib
    #used for jupyter notebook matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    for tm in times:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        df[df.hr == tm].plot(kind = 'line', 
                              x = 'days', y = 'cnt', ax = ax)          
        df[df.hr == tm].plot(kind = 'line', 
                              x = 'days', y = 'predicted', color = 'red', ax = ax)                                    
        plt.xlabel("Days from start")
        plt.ylabel("Number of bikes rented")
        plt.title("Bikes rented for hour = " + str(tm))
        fig.savefig('ts_' + str(tm) + '.png')
    plt.show()
    return 'Done'

def resids(df):
    df['resids'] = df.predicted - df.cnt
    return df        
        
def box_resids(df):
    import matplotlib
    #used for jupyter notebook matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    
    df = resids(df)
    
    fig = plt.figure(figsize=(12, 6))
    fig.clf()
    ax = fig.gca()  
    df.boxplot(column = ['resids'], by = ['workHr'], ax = ax)   
    plt.xlabel('')
    plt.ylabel('Residuals')
    fig.savefig('boxes' + '.png')
    plt.show()
    return 'Done'
    
 
def ts_resids_hist(df, times):
    import matplotlib
    #used for jupyter notebook matplotlib.use('agg')  # Set backend
    import matplotlib.pyplot as plt
    for tm in times:
        fig = plt.figure(figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        ax.hist(df.loc[df.hr == tm, 'resids'], bins = 30)
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title("Histograms of residuals for hour = " + str(tm))
        fig.savefig('hist_' + str(tm) + '.png')
    plt.show()
       
    return 'Done'

#MAKE SURE TO RUN GRAPHS WITH X_TEST which is the predicted DF

X_test['cnt'] = y_test
X_test = X_test.sort_values(by = ['days', 'hr'], axis = 0, ascending = True)
times = [6, 8, 10, 12, 14, 16, 18, 20, 22]
''' USED FOR JUYPTER NOTEBOOK AND AZURE ML
def azureml_main(df):
    df = df.sort(['days', 'hr'], axis = 0, ascending = True)
    times = [6, 8, 10, 12, 14, 16, 18, 20, 22]
    ts_bikes(df, times)
    box_resids(df)
    ts_resids_hist(df, times)
    return df
'''
   
#BONUS SEPERATING WORKING HOURS FROM WEEKEND AND HOLIDAY HOURS, ERASING THE LOWER 20% QUANTILE OF COUNT DATA
work_day = df['workingday']
holiday = df['holiday']
df['isWorking'] = np.where(np.logical_and(work_day == 1, holiday == 0),1,0)

#add a total month count from the start of the dataset to the end
df['monthCount'] = (12 * df.yr + df.mnth)

isWorking = df['isWorking'].to_numpy()
df['workHr'] = np.where(isWorking, df.hr, df.hr + 24.0)
df = df.drop(['workingday', 'holiday', 'hr'], axis =1)


#create a quantile based on 20% of the data
out = df.groupby(['monthCount', 'workHr']).cnt.quantile(q = 0.2)
out = pd.DataFrame(out)
out.reset_index(inplace= True)
out.columns = ['monthCount', 'workHr', 'quantile']
#save columns names
in_names = list(df)
#merge quantiles to the DF
df = pd.merge(df, out, left_on = ['monthCount', 'workHr'], right_on = ['monthCount', 'workHr'], how = 'inner')
#if the Count is > Quantile keep it otherwise erase it
df = df.loc[(df['cnt'] > df['quantile']), :]
#fix the dataset
df = df.drop(['quantile'], axis = 1)
df.columns = in_names
df = df.sort_values(by = ['days', 'workHr'], axis = 0, ascending = True)

#new statistics on changed dataset
y = df['cnt']
X = df.drop(['cnt', 'dayOfWeek'], axis = 1)
#split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

#linear regression model and score model

model = LinearRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('r2', score)
y_pred = model.predict(X_test)
y_mean = []
for i in y_pred:
    y_mean.append(189.46)
y_mean = np.array(y_mean)

#appended column cnt mean to all predicted guesses to use for simple predictor 
#RELATIVE SQUARED ERROR and ROOT MEAN SQUARED ERROR
simple_pred = y_mean
squared_error = mean_squared_error(y_test, y_pred)
root_mean_squared = squared_error ** .5
simple_pred_error = mean_squared_error(y_test, y_mean)

#prediction model squared error / simple predictor squared error = relative square error
relative_squared_error = squared_error / simple_pred_error

print('root mean square', root_mean_squared)
print('relative squared error', relative_squared_error)

#mean absolute error
absolute_square_error = mean_absolute_error(y_test, y_pred)

print('mean absolute error', absolute_square_error)
#relative absolute error = absolute_square error / root_mean squared
relative_absolute_error = absolute_square_error / root_mean_squared
#Relative absolute error (RAE) is the relative absolute difference between expected
#and actual values; relative because the mean difference is divided by the arithmetic mean.
#Not sure which equation is correct for relative absolute error
#Don't remember how I got 189.46 for the mean
print('relative absolute error', relative_absolute_error)
print('test relative absolute error', absolute_square_error / 189.46)
'''
#SETUP X_test for visualizations
X_test['cnt'] = y_test
X_test = X_test.sort_values(by = ['days', 'workHr'], axis = 0, ascending = True)
X_test['predicted'] = y_pred
#box_resids(X_test)
'''

print('DECISION FOREST REGRESSION STATS')
#Decision Forest Regression
model = RandomForestRegressor(max_depth=32, random_state=0, n_estimators=8, oob_score = False)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('r2', score)
y_pred = model.predict(X_test)

#appended column cnt mean to all predicted guesses to use for simple predictor 
#RELATIVE SQUARED ERROR and ROOT MEAN SQUARED ERROR
simple_pred = y_mean
squared_error = mean_squared_error(y_test, y_pred)
root_mean_squared = squared_error ** .5
simple_pred_error = mean_squared_error(y_test, y_mean)

#prediction model squared error / simple predictor squared error = relative square error
relative_squared_error = squared_error / simple_pred_error

print('root mean square', root_mean_squared)
print('relative squared error', relative_squared_error)

#mean absolute error
absolute_square_error = mean_absolute_error(y_test, y_pred)

print('mean absolute error', absolute_square_error)
#relative absolute error = absolute_square error / root_mean squared
relative_absolute_error = absolute_square_error / root_mean_squared
#Relative absolute error (RAE) is the relative absolute difference between expected
#and actual values; relative because the mean difference is divided by the arithmetic mean.
#Not sure which equation is correct for relative absolute error
#211.644758 is the mean using df.mean() for the CNT column
print('relative absolute error', relative_absolute_error)
print('test relative absolute error', absolute_square_error / 211.644758)

#test relative absoulte error matches the AZURE ML EVALUATE MODEL better
