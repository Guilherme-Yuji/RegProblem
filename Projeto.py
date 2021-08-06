#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


'''
---------- Data Importing ----------
'''
#Reading the first dataset
data_1 = pd.read_csv('md_raw_dataset.csv',sep = ';')

#Showing the dataset
print(data_1)

#Reading the second dataset
data_2 = pd.read_csv('md_target_dataset.csv', sep = ';')

#Showing the dataset
print(data_2)

#Renaming the column name of the second dataset
data_2 = data_2.rename(columns={'groups':'groups_target'})

#Concatenating the two datasets
data = pd.concat([data_1,data_2], axis = 1)

#Display more columns in the print function
pd.set_option('display.max_columns', 50)

#Getting the shape of the final dataset
print(data.shape)

#Showing the first 5 rows
print(data.head())

#Showing the last 5 rows
print(data.tail())

'''
As the dataset 2 has 3 rows less than the dataset 1, in the final dataset 
is missing 3 values of the target variable, so these rows will be removed.
'''
data = data.iloc[:-3]


'''
---------- Data Treatment ----------
'''

#Printing information of the type of the data and missing values
print(data.info())

'''
There are some columns with date information, so they will be put 
in a list to be easier to convert to the correct type
'''
date_columns = ['when','expected_start','start_process','start_subprocess1','start_critical_subprocess1','predicted_process_end','process_end','subprocess1_end','reported_on_tower','opened']

'''
As the opened column is contains date information and the first 3 lines don't have
this type of information, they will be replace with the value zero because we don't know
the correct date.
'''
data['opened'] = data['opened'].replace(['44021.58091','44021.6737','44021.70867'],'0')

#Converting the date columns to the correct type
for col in date_columns:
    data[col] = pd.to_datetime(data[col], errors='coerce')

#Getting the numerical columns
numerical_columns = data.select_dtypes(['float64','int64']).columns

#Getting the categorical columns
categorical_columns = data.select_dtypes('object').columns

#Getting the number of unique values in the categorical columns
for col in categorical_columns:
    print(col,':',data[col].nunique(), 'unique values.')

'''
As we can observe in the result, the column 'etherium_before_start'
has too many unique values, so probably there are numerical and pandas
could not interprete the right type due to some problems
'''

#Searching for the value that isn't numerical
wrong_data_index = []
for i in range(len(data['etherium_before_start'])):
    try:
        float(data['etherium_before_start'][i])
    except:
        wrong_data_index.append(i)
        

#Printting the problem index
print(wrong_data_index)

#As we can see, just one row has a non numerical value, so let's print it.
print(data['etherium_before_start'][7829])

#As we don't know the correct value for this row, it will be replaced with zero.
data['etherium_before_start'] = data['etherium_before_start'].replace('21/12/2020 12:11',0)

#Converting this column to numerical type
data['etherium_before_start'] = data['etherium_before_start'].apply(float)

#Redefining the numerical_columns variable due to this change
numerical_columns = data.select_dtypes(['float64','int64']).columns

#Redefining the categorical_columns variable due to this change
categorical_columns = data.select_dtypes('object').columns

#Checking if these 3 columns are representing all the columns of the dataset
print(len(numerical_columns)+len(categorical_columns)+len(date_columns))

#Looking for null values
print(data.isnull().sum())

#Observing the distribution of the data throught descriptive statistics
print(data.describe())

'''
As we have a considerable number of null values and the variables have a
significant value, the best option is to replace the null values by the
mean of the variable to get closer to real values.
'''
for col in numerical_columns:
    data[col] = data[col].fillna(data[col].mean())

#Checking if the if all the null values in the numerical columns were solved.
print(data[numerical_columns].isnull().sum())

#Checking if there are missing values in the categorical columns
print(data[categorical_columns].isnull().sum())

#Checking if there are missing values in the date columns
print(data[date_columns].isnull().sum())
'''
As shown, there are null values in the date columns but as they won't be used in
the machine learning algorithm and we don't know the correct value, they are no need
to solve them.
'''

#Printing the unique values in each categorical column and it frequency.
for col in categorical_columns:
    print(data[col].value_counts())
    print()

'''
As we can seem there are some value with strange information. Probably it is
miss typed information. So they will be corrected.
'''

#Correcting the values
data['super_hero_group'] = data['super_hero_group'].replace('₢','C')
data['crystal_supergroup'] = data['crystal_supergroup'].replace('1ª','1')
data['Cycle'] = data['Cycle'].replace('33','3ª')


'''
---------- EDA ----------
'''

#Setting the style of the plots
sns.set_style('darkgrid')


#Looking for outliers in the numerical columns with boxplot and saving them as image
for col in numerical_columns:
    ax = sns.boxplot(data[col])
    #plt.show()
    figure = ax.get_figure()
    name = col.split(sep=':')
    name = '_'.join(name)
    figure.savefig('boxplot/'+name+'.png')
    plt.close()

'''
The data contains outliers in some columns but as we don't if they are mistakes
or real value, the best action is to keep them in the data.
'''

#Getting descriptive statistics
print(data.describe())

#Ploting the relationship of the numerical values and the target variable
for col in numerical_columns[:-1]:
    plt.figure(figsize = (15,5))
    ax = sns.scatterplot(x = data[col], y = data['target'])
    #plt.show()
    figure = ax.get_figure()
    name = col.split(sep=':')
    name = '_'.join(name)
    figure.savefig('scatterplot/'+name+'.png')
    plt.close()


#Plotting the correlation between the numerical variables
plt.figure(figsize = (20,20))
ax = sns.heatmap(data.corr(), annot = True)
#plt.show()
figure = ax.get_figure()
figure.savefig('Correlation.png')
plt.close()

#Plotting the categorical variables and its frequencies
for col in categorical_columns:
    if data[col].nunique() <= 10:
        total = len(data[col])
        plt.figure(figsize = (15,5))
        ax = sns.countplot(x = data[col], order = data[col].value_counts().index)
        for p in ax.patches:
            height = p.get_height()
            ax.text(x = p.get_x()+(p.get_width()/2), y = height * 1.01 , s = '{:.2f}%'.format(height/total*100), ha = 'center')
        #plt.show()
        figure = ax.get_figure()
        figure.savefig('categorical/'+col+'.png')
        plt.close()


#Plotting the relashionship between categorical variables and the target with boxenplot
for col in categorical_columns:
    if data[col].nunique() <= 10:
        plt.figure(figsize = (15,5))
        ax = sns.boxenplot(x = data['target'], y = data[col])
        #plt.show()
        figure = ax.get_figure()
        figure.savefig('boxenplot/'+col+'.png')
        plt.close()

#Plotting the relashionship between categorical variables and the target with histograms
for col in categorical_columns:
    if data[col].nunique() <= 10:
        plt.figure(figsize = (15,5))
        ax = sns.histplot(data = data , x = 'target', hue = col, kde = True)
        #plt.show()
        figure = ax.get_figure()
        figure.savefig('histogram/'+col+'.png')
        plt.close()

#Checking the crystal type column values
for col in categorical_columns:
    print(data[col].value_counts())
    print()


'''
---------- Feature Engineering ----------
'''

#Creating the x and y variables
#As the date columns don't have useful information for the ML algorithms, 
#they will be take out from x variable
x = data.drop(date_columns, axis = 1).drop('target',axis = 1)
y = data['target']

#Starting the encoder for the categorical columns
cat_enc = TargetEncoder(cols = categorical_columns).fit(x,y)

#Encoding the categorical varibles
x_encoded = cat_enc.transform(x)

#Checking the x encoded variables
print(x_encoded)


'''
---------- Machine Learning ----------
'''

'''
As request from the challenge, it's needed to choose one machine learning algorithm. So I will choose
Random Forest because it is a very strong algorithm that use ensemble techniques to improve the 
performance.

'''

#Starting the ML Model
model = RandomForestRegressor()


#Validating the model using Cross-validation
'''
As its needed to choose a metrics, i chose RMSE because it is a good metrics and has the same
unit of the target variable, so it's easier to interpretate.
'''
rmse = cross_val_score(model,x_encoded,y,cv = 10, scoring = 'neg_root_mean_squared_error' )

#Printing the mean value of RMSE
print(np.abs(rmse.mean()))

'''
With a mean value of approximately 23.7 of the metrics Root Mean Squared Error and a 
mean value of the target of 99.7, it is possible to conclude that the algorithm has
an error of 24% in predicting new values.
'''

#Training the model
model_1 = RandomForestRegressor()
model_1.fit(x_encoded,y)

#Getting the most important variables to predict the target variable
importance = pd.DataFrame({'Variables': x_encoded.columns, 'Importance': model_1.feature_importances_})

#Showing the top 10 important variables
print(importance.sort_values('Importance', ascending = False).head(10))

#Plotting the importance
plt.figure(figsize = (15,8))
ax = sns.barplot(y = importance['Variables'], x = importance['Importance'], order = importance.sort_values('Importance', ascending = False)['Variables'])
#plt.show()
figure = ax.get_figure()
figure.savefig('feature_importancy.png')
plt.close()

#Predicting New values
#y_pred = model_1.predict([[VALUES]])