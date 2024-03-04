import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('brightondata.csv')

data = data.dropna()
X = data[['Year', 'Median Sales Price', 'Total Population', 'Median Income Level']]
y = data['Total Housing Units']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)

coefficients = model.coef_
intercept = model.intercept_
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

formula = f"Total Housing Units = {intercept:.2f} + {coefficients[0]:.2f}*Year + {coefficients[1]:.2f}*Median Sales Price + {coefficients[2]:.2f}*Total Population + + {coefficients[3]:.2f}*Median Income Level"
print("Linear Regression Formula:")
print(formula)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
