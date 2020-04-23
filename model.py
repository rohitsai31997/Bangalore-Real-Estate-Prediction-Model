import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)

df1 = pd.read_csv(r"C:\Users\rohit\Desktop\Datasets\Bengaluru_House_Data.csv")
#Gives the number of rows and columns
# print(df1.shape)

#To get the number of rows in each area type and number of bed rooms
# print(df1.groupby('area_type')['area_type'].agg('count'))
# print(df1.groupby('size')['size'].agg('count'))

#Dropping some columns as we assume they're not that important
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis = 'columns')
# print(df2.head())

#Shows the number of null values in the remaining features
# print(df2.isnull().sum())
#Dropping the rows which have null values
df3 = df2.dropna()
# print(df3.isnull().sum())

"""The number of bedrooms column is filled by many people in 
many different ways. For ex. 1BHK , 1 Bedroom etc. So we 
have to convert them all into the same format."""
#Checking the number of unique values that number of bedrooms has
# print(df3['size'].unique())

#Creating a new column 'bhk' using the 'size' column
#Taking the first token of every row in 'size' column
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
#print(df3['bhk'].unique())

#Outliers with more than 20 bedrooms
#print(df3[df3.bhk>20])

#We have ranges in sqft feature. We have convert it to a single int value
#print(df3.total_sqft.unique())

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

#Prints all the houses which have the sqft in ranges
#print(df3[~df3['total_sqft'].apply(is_float)])

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
#Example call to the function - Returns the avg of the range
#print(convert_sqft_to_num('2100 - 2850'))

df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
# print(df4.head(3))
#To find the specific index
# print(df4.loc[30])

df5 = df4.copy()
#Creating a feature called 'price per sq.ft'
df5['price_per_sqft'] =  df5['price']*100000/df5['total_sqft']
#print(df5.head())

# Number of unique locations
# print(len(df5.location.unique()))

#Stripping the extra spaces in the location feature
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
# print(location_stats)

#Setting a threshold for the number of rows in a location
# print(len(location_stats[location_stats<=10]))

location_stats_less_than_10 = location_stats[location_stats<=10]
# print(location_stats_less_than_10)

# print(len(df5.location.unique()))

#Only taking the locations that have more than 10 rows of data
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
# print(len(df5.location.unique()))

# print(df5.head(10))

#Outlier Detection

"""Assuming that each bedroom should be atleast 300sq.ft, we
can print all the rows of data that don't satisfy this criteria"""
#These are anomalies or outliers. So we can safely remove them
#print(df5[df5.total_sqft/df5.bhk <300].head())

#removing those outliers
# print(df5.shape)
df6 = df5[~(df5.total_sqft/df5.bhk <300)]
# print(df6.shape)

#Another type of outliers can be in price_per_sq_ft - Too high or two low
# print(df6.price_per_sqft.describe())

#Now, we have to remove the extreme cases
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

# print(df6.shape)
df7 = remove_pps_outliers(df6)
# print(df7.shape)

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft, bhk2.price, color = 'blue', label = '2 BHK', s = 50)
    plt.scatter(bhk3.total_sqft, bhk3.price, marker="+", color = "green", label = "3 BHK", s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price Per Square Feet")
    plt.title(location)
    plt.legend()

# plot_scatter_chart(df7, "Rajaji Nagar")

"""We should also remove properties where, for the same location, 
the price of 3BHK is less than 2 bhk apts with the same sqft area
What we will do is , for a given location, we will build a dictionary of stats 
per bhk ie
{
    '1' : {
            'mean' : 4000
            'std'   : 2000,
            'count' : 34
        }
    '2' : {
            'mean' : 4300
            'std'   : 2300,
            'count' : 22
        }
    
    Now, we can remove those 2 BHK apartments, whose price_per_sqft
    is less than the mean price_per_sqft of 1 BHK apartment
        """
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                'mean' : np.mean(bhk_df.price_per_sqft),
                'std' : np.std(bhk_df.price_per_sqft),
                'count' : bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft< (stats['mean'])].index.values)
    return df.drop(exclude_indices,axis = 'index')

# print(df7.shape)
df8 = remove_bhk_outliers(df7)
# print(df8.shape)

#to check the scatter plot to see if the most of the outliers are removed
# plot_scatter_chart(df8, "Hebbal")

#Plotting a histogram to see how many buildings are in different categories of price_per_sqft
import matplotlib
# matplotlib.rcParams["figure.figsize"] = (20,10)
# plt.hist(df8.price_per_sqft, rwidth=0.8)
# plt.xlabel("Price Per Square Feet")
# plt.ylabel("Count")

#Now check the bathroom outliers
# print(df8.bath.unique())


# print(df8[df8.bath>10])

"""Assume that if the number of baths is greater than the number of
bedrooms + 2, then wwe remove all those points"""

#Histogram of number of baths
# plt.hist(df8.bath , rwidth=0.8)
# plt.xlabel("Number of Bathrooms")
# plt.ylabel("Count")

#Checking the datapoints with baths > bedrooms+2
# print(df8[df8.bath> df8.bhk+2])

#Removing the above outliers
# print(df8.shape)
df9 = df8[df8.bath < df8.bhk+2]
# print(df9.shape)

"""The dataset df9 looks neat now. We can start preparing 
it for ML training. So we have to drop some unnecessary features 
Price_per_sqft and size are not that useful for now"""

df10 = df9.drop(['size', 'price_per_sqft'], axis='columns')
# print(df10.head(3))

#MODEL BUILDING
#One hot  encoding on location
dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis = 'columns')], axis='columns')

#As we already used one hot encoding for that feature
df12 = df11.drop('location', axis='columns')

# print(df12.shape)

X = df12.drop('price', axis= 'columns')
# print(X.head())

y = df12.price
# print(y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
# print(lr_clf.score(X_test, y_test))

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
# print(cross_val_score(LinearRegression(), X, y, cv=cv))

#Grid Search CV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params': {
                'normalize' : [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha' : [1,2],
                'selection' : ['random', 'cyclic']
            }
        },
        'decision_tree':{
            'model' : DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse', 'friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'],cv=cv, return_train_score = False)
        gs.fit(X, y)
        scores.append({
            'model' : algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })

    return pd.DataFrame(scores, columns = ['model','best_score', 'best_params'])

#Lr_regression gives the best scores
print(find_best_model_using_gridsearchcv(X,y))

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return lr_clf.predict([x])[0]

# print(predict_price("1st Phase JP Nagar", 1000,3,3))

"""The model is ready. Now we create  a file for the model 
and dump the model onto the file"""

import pickle
with open('bangalore_home_prices_model.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json", 'w') as f:
    f.write(json.dumps(columns))