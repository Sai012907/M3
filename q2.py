import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

homeless_population = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
housing_units = np.array([1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000])

X = housing_units.reshape(-1, 1)
y = homeless_population

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

future_housing_units = np.array([2100, 2200, 2300]).reshape(-1, 1)
future_homelessness = model.predict(future_housing_units)

for year, prediction in zip([2055, 2060, 2065], future_homelessness):
    print(f'Year {year}: Predicted Homeless Population - {round(prediction)}')
