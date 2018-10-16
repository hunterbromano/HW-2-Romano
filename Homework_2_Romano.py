
# coding: utf-8

# # Homework 2 | Hunter Romano | Housing Price Predictor

# In[417]:


# The goal of this assignment is to predict the sales price for each house 
# root mean squared error is the evaluation metric. 


# # Part 1 | Import Packages & the Data Set

# In[418]:


# package for listing and algebra
import numpy as np

# package for data processing
import pandas as pd

# package for plots
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# package for regression analysis
import statsmodels.api as sm
from math import sqrt
import os

# package for statistics

from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p

# package for more plotting
import seaborn as sns

# import a specific color palette and style guide
color = sns.color_palette()
sns.set_style('darkgrid')

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, cross_val_score

from IPython.display import display, FileLink

# Cool command to get rid of useless warning messages
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn


# In[419]:


# import the data set | 
house = pd.read_csv("/Users/hunterromano/Desktop/train.csv")

# look at data to make sure it was imported and correct
house.head(5)


# In[420]:


# I want a full list of the attributes in the set
# I can understand what data points I have and how to move forward
house.dtypes


# In[421]:


# I want a better understanding of the values 
house.describe()


# In[422]:


# More views
house.head().transpose()


# # Part 2 | Clean & Explore the Data

# In[423]:


# First I want to explore outliers, 
#as it was the first cleaning we covered in class
# https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
# ^ This kernel helped me learn a lot about cleaning data 
# the dataset page says that a plot of sale price and gr liv area will quickly help
# someoen determine 5 data points that should be removed, so let's do that
fig, outlier_discovery = plt.subplots()
outlier_discovery.scatter(x=house['GrLivArea'], y=house['SalePrice'])
plt.ylabel('Sale Price', fontsize=15)
plt.xlabel('Ground Living Area' , fontsize=15)
plt.title ('Ground Floor Space Related to Price', fontsize = 18)
plt.show()


# In[424]:


# The points to the far bottom right are obvious outliers
# I will delte them in order to not skew the data
#house = house.drop(house[(house['GrLivArea']>4000) & (house['SalePrice']<300000)].index)

# the data page says 5 observations though so I assume I shoudl remove those at the far upper right as well
house = house.drop(house[(house['GrLivArea']>4000)].index)


# In[425]:


# Check to make sure that they are gone
fig, outlier_discovery = plt.subplots()
outlier_discovery.scatter(x=house['GrLivArea'], y=house['SalePrice'])
plt.ylabel('Sale Price', fontsize=15)
plt.xlabel('Ground Living Area' , fontsize=15)
plt.title ('Ground Floor Space Related to Price', fontsize = 18)
plt.show()


# In[426]:


# Intuitevily one would think that total square feet is a major precitor of price
# Referring to prior lists this attribute doesn't exist, so I will create it
house['total_sf'] = (house['BsmtFinSF1'].fillna(0) + house['BsmtFinSF2'].fillna(0) + house['1stFlrSF'].fillna(0) + house['2ndFlrSF'].fillna(0))


# In[427]:


# I want to make sure that worked, so let's take a look
house.total_sf.head()


# In[428]:


# This competition requires Sale Price predictions
# As the target variable I will remove from the data frame and take the log
# I like the pop removal but was having trouble with my columns so I will keep it there just after a hashtag more as a note for myself
sale_price = house.pop('SalePrice')
sale_price_log = np.log(sale_price)
#house.drop(['SalePrice'], axis=1, inplace=True)


# In[429]:


# This variable is numerical but doesn't mean anything, i will keep pop command though even behind hashtag just for reference
#house_id = house.pop = ('Id')
house.drop(['Id'], axis=1, inplace=True)


# In[430]:


# As I cntinue to clean I need to divide attribute columns between 
# those that are continuous and those that are categorical
# continuous contains numerical data points 
# categorical contains text data points
# I will divide and label them based on the type of data in them
house.dtypes


# In[431]:


# These attributes contain numerical data
continuous_data = [
    'BsmtUnfSF',
    'FullBath',
    'LotFrontage',
    'BsmtFullBath',
    '3SsnPorch',
    'BedroomAbvGr',
    'LowQualFinSF',
    'BsmtFinSF1',
    'WoodDeckSF',
    'GarageArea',
    'MiscVal',
    'BsmtHalfBath',
    'HalfBath',
    'EnclosedPorch',
    'ScreenPorch',
    'TotRmsAbvGrd',
    'Fireplaces',
    'KitchenAbvGr',
    'GarageCars',
    '1stFlrSF',
    'BsmtFinSF2',
    'PoolArea',
    '2ndFlrSF',
    'TotalBsmtSF',
    'total_sf',
    'GrLivArea',
    'LotArea',
    'OpenPorchSF',
    'MasVnrArea'
]


# In[432]:


# By default the other attributes are categorical
categorical_data = [col for col in house.columns if col not in continuous_data]

# check to make sure I did that correctly
categorical_data


# In[433]:


# Check this to make sure all is well

assert len(house.columns) == len(categorical_data + continuous_data)


# In[434]:


# Now I need to convert categorical columns into a pandas datatype
for col_name, col in house[categorical_data].items():
    house[col_name] = col.astype('category').cat.as_ordered()


# In[435]:


# I need to now ensure that order type attributes are ordered correctly.
ordinal_data =[
    ('ExterQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('ExterCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtExposure', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('BsmtFinType1', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('BsmtFinType2', ['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']),
    ('HeatingQC', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('KitchenQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('FireplaceQu', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageFinish', ['Unf', 'Rfn', 'Fin']),
    ('GarageQual', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('GarageCond', ['Po', 'Fa', 'TA', 'Gd', 'Ex']),
    ('PoolQC', ['Fa', 'TA', 'Gd', 'Ex']),
    ('OverallQual', list(range(1, 11))),
    ('OverallCond', list(range(1, 11))),
    ('LandSlope', ['Sev', 'Mod', 'Gtl']),
    ('Functional', ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ']),
    ('YearBuilt', list(range(1800, 2018))),
    ('YrSold', list(range(2006, 2018))),
    ('GarageYrBlt', list(range(1900, 2018))),
    ('YearRemodAdd', list(range(1900, 2018)))
]

ordinal_columns = [o[0] for o in ordinal_data]

for col, categories in ordinal_data:
    house[col].cat.set_categories(categories, ordered = True, inplace=True)
   


# In[436]:


#Conner and I discussed pd.factorize which I tried as well, but found a more intuitive version of this method online

# Now to deal with columns with no ordinal relationship
other_columns = [col for col in categorical_data if col not in ordinal_columns]

assert len(categorical_data) == len(ordinal_columns + other_columns)


# # Replacing Missing Data

# In[437]:


# Now that we have identified holes in the data we need to fill them
# I will be replacing these holes with median values
NAs = {}

for col in (
 'GarageArea', 'GarageCars', 'BsmtFinSF1',
    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
    'MasVnrArea'):
    
    NAs[col] = 0
    house[col] = house[col].fillna(0)
    house[f'{col}_na'] = pd.isna(house[col])


# In[438]:


# Now to work on the continuous type variables
for col in continuous_data:
    if not len(house[house[col].isna()]):
        continue
        
    median = house[col].median()
    
    house[f'{col}_na'] = pd.isna(house[col])
    house[col] = house[col].fillna(median)
    
    NAs[col] = median    


# In[439]:


# Now to unskew the data
# Let us use Scikit Learn to visualize the data disctribution

skew_feats = house[continuous_data].apply(skew).sort_values(ascending=False)
skew_feats.head(20)

# looks good to me


# In[440]:


# The most skewed attribute is MiscVal
sns.distplot(house[house['MiscVal'] !=0]['MiscVal'])
plt.title ('MiscVal Disctribution', fontsize = 18)
plt.show()


# In[441]:


# By logging the most skewed attributes, the overall data quality should improve
skew_feats = skew_feats[abs(skew_feats) > .75]

for feat in skew_feats.index:
    house[feat] = np.log1p(house[feat])
    


# In[442]:


# Look back to see how that disctribution has changed
sns.distplot(house[house['MiscVal'] !=0]['MiscVal'])
plt.title ('MiscVal Disctribution', fontsize = 18)
plt.show()


# # Add numbers and dummies

# In[443]:


# In order to predict price we need to create dummies to replace text attribute values
house_numbered = house.copy()
dummies = pd.get_dummies(house_numbered[other_columns], dummy_na=True)
for col_name in categorical_data:
    # Add 1 to replace values of -1 with 0
    house_numbered[col_name] = house_numbered[col_name].cat.codes +1
    


# In[444]:


# Finalize the dummie assignments
# Drop the useless values
house_numbered.drop(other_columns, axis=1, inplace=True)
house_numbered = pd.concat([house_numbered, dummies], axis=1)


# # Train Models

# In[445]:


# My colleague at work, works with machine learning
# After several models that yielded poor results he recomended KFolds cross-validatin, and Lasso
# The Lasso model should help with the overfitting of a straight linear regression
kf = KFold(n_splits=10, shuffle=True, random_state=42)

model = Lasso(alpha=0.0004)

scores = np.sqrt(-cross_val_score(model, house_numbered, sale_price_log, cv=kf, scoring='neg_mean_squared_error'))


# In[446]:


# Mean looks great so now let's train
scores.mean()


# In[447]:


train_model = Lasso(alpha=0.0004)
train_model.fit(house_numbered, sale_price_log)


# In[449]:


# Model is all set so let's run this with our testing data
house_test = pd.read_csv("/Users/hunterromano/Desktop/test.csv")


# In[451]:


# Put this data set through the exact same steps we put our training data through
house_test.drop(['Id'], axis=1, inplace=True)
#house_id = house_test.pop('Id')


# In[452]:


house_test['total_sf'] = (
    house_test['BsmtFinSF1'].fillna(0) + house_test['BsmtFinSF2'].fillna(0) +
    house_test['1stFlrSF'].fillna(0) + house_test['2ndFlrSF'].fillna(0))


# In[453]:


house_numbered.describe()


# In[454]:


len(categorical_data)


# In[455]:


house.describe()


# In[456]:


house_test.describe()


# In[457]:


#for col_name in categorical_data:
   # house_test[col_name] = (
      #  pd.Categorical(
         #   house_test[col_name], 
         #   categories=house[col_name].cat.categories, 
          #  ordered=True))


# In[458]:


for col in continuous_data:
    if col not in NAs: continue
        
    house_test[f'{col}_na'] = pd.isna(house_test[col])
    house_test[col] = house_test[col].fillna(NAs[col])


# In[459]:


# Control for remaining NAs
house_test[continuous_data] = house_test[continuous_data].fillna(house_test[continuous_data].median())


# In[460]:


for feat in skew_feats.index:
    house_test[feat] = np.log1p(house_test[feat])


# In[461]:


house_final_test = house_test.copy()


# In[462]:


test_dummies = pd.get_dummies(house_final_test[other_columns], dummy_na=True)
for col_name in categorical_data:
    house_final_test[col_name] = house_final_test[col_name].cat.codes + 1
house_final_test.drop(other_columns, axis=1, inplace=True)
house_final_test = pd.concat([house_final_test, test_dummies], axis=1)


# In[391]:


#predictions = train_model.predict(house_final_test)


# In[392]:


# Create CSV and take the reverse log
#pd.DataFrame({'Id': house_id, 'SalePrice': np.exp(predictions)}).to_csv('output.csv')


# In[416]:


categorical_data

