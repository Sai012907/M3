import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Sample data representing the past homeless population for a region
years = np.array([2010, 2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050])
homeless_population = np.array([500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])

# Create a pandas DataFrame
data = pd.DataFrame({'Year': years, 'Homeless_Population': homeless_population})
data.set_index('Year', inplace=True)

# Fit ARIMA model
model = ARIMA(data, order=(1, 1, 1))  # Adjust order based on your data characteristics
fit_model = model.fit()

# Forecast future homeless population
future_years = np.array([2060, 2070, 2090])
forecast, stderr, conf_int = fit_model.forecast(steps=len(future_years))

# Plotting the results
plt.plot(data.index, data['Homeless_Population'], label='Actual Data')
plt.plot(np.append(data.index, future_years), np.append(data['Homeless_Population'], forecast), linestyle='dashed', color='red', label='ARIMA Forecast')
plt.title('Homeless Population Prediction using ARIMA')
plt.xlabel('Year')
plt.ylabel('Homeless Population')
plt.legend()
plt.show()

# Display the forecasts for the future years
for year, prediction in zip(future_years, forecast):
    print(f'Year {year}: Predicted Homeless Population - {round(prediction)}')

