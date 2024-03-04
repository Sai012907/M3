import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('manchesterdata.csv')
data = data[['Total Housing Units', 'Homelessness']]
data = data.dropna()

X = data[['Total Housing Units']]
y = data['Homelessness']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

future_housing_units = np.array([280607, 322420, 447862]).reshape(-1, 1)
future_homelessness = model.predict(future_housing_units)

for year, prediction in zip([2034, 2044, 2074], future_homelessness):
    print(f'Year {year}: Predicted Homeless Population - {round(prediction)}')
