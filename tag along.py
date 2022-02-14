import pandas as pd
import numpy as np

df = pd.read_csv('energydata_complete.csv')
corr = pd.DataFrame(df.corr())

# question 12
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df['T2']
X = np.array(X).reshape(-1,1)

y = df['T6'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

predicted_values = linear_regressor.predict(x_test)

from sklearn.metrics import r2_score

Rsquared = round(r2_score(y_test, predicted_values), 2)
# R-squared = 0.65

# question 13

from sklearn.preprocessing import MinMaxScaler

df = df.drop(['date', 'lights'], axis=1)
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
X = normalized_df.drop('Appliances', axis=1)
y = normalized_df['Appliances']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mult_regressor = LinearRegression()
mult_regressor.fit(x_train, y_train)
predicted = mult_regressor.predict(x_test)

from sklearn.metrics import mean_absolute_error

mae = round(mean_absolute_error(y_test, predicted), 2)
# mae = 0.05

# question 14

from sklearn.metrics import mean_squared_error

mse = round(mean_squared_error(y_test, predicted), 2)
# mse = 0.01

# 15

from math import sqrt
mse = mean_squared_error(y_test, predicted)
rmse = round(sqrt(mse), 3)

# question 16
determination = round(r2_score(y_test, predicted), 2)

# question 17

weights = pd.Series(mult_regressor.coef_, x_train.columns).sort_values()

# question 18

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=0.4)
ridge_reg.fit(x_train, y_train)

ridge_prediction = ridge_reg.predict(x_test)
r_mse = mean_squared_error(y_test, ridge_prediction)
rmse = round(sqrt(r_mse), 3)

# question 19

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.001)
lasso_reg.fit(x_train, y_train)
lasso_prediction = lasso_reg.predict(x_test)

l_weights = pd.Series(lasso_reg.coef_, x_train.columns).sort_values()

# question 20
mse = mean_squared_error(y_test, lasso_prediction)
lrmse = round(sqrt(mse), 3)

