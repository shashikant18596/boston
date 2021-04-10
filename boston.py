import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.datasets import load_boston
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns',None)
boston = load_boston()
#print(type(boston))
print(boston.keys())
#print(len(boston.target))
df = pd.DataFrame(data=boston.data,columns=boston.feature_names)
#print(df)

print(df.describe())

'''for columns in df.columns:
    sns.displot(df[columns],kde=True)
plt.show()'''

# checking for outliers

'''for columns in df.columns:
    sns.boxplot(data=df[df.columns])
plt.show()'''

q = df['CRIM'].quantile(0.99)
data_cleaned= df[df['CRIM']<q]

q = df['ZN'].quantile(0.99)
data_cleaned= df[df['ZN']<q]

q = df['B'].quantile(0.99)
data_cleaned= df[df['B']<q]

q = df['LSTAT'].quantile(0.99)
data_cleaned= df[df['LSTAT']<q]

'''for columns in df.columns:
    sns.boxplot(data=df[df.columns])
plt.show()'''

scalar = StandardScaler()
x_scaled = scalar.fit_transform(df[df.columns])
y = pd.DataFrame(data=boston.target)
print(y.shape)

vif = pd.DataFrame()
vif['vif'] = [variance_inflation_factor(x_scaled,i)for i in range (x_scaled.shape[1])]
print(vif)

x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.25)
model = LinearRegression()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
print()










