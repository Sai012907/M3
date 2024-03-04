import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing


data = pd.read_csv('brightondata.csv')

data = data.dropna()
data = data[['Year', 'Total Housing Units']]

# Perform Double Exponential Smoothing
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Set 'Year' as the index
data.set_index('Year', inplace=True)

# Perform Double Exponential Smoothing
model = ExponentialSmoothing(data['Total Housing Units'], trend='add', seasonal=None)
fit_model = model.fit()

# Make predictions for future years
future_years = pd.date_range(start=data.index[-1] + pd.DateOffset(years=1), periods=50, freq='A')  # Adjust as needed
future_predictions = fit_model.predict(start=len(data), end=len(data) + len(future_years) - 1)
for year in [10, 20, 50]:
    print(f'Year {year}: Predicted Housing Supply - {int(future_predictions[year-1])}')
# Plotting the results
plt.plot(data.index, data['Total Housing Units'], label='Actual Housing Supply')
plt.plot(future_years, future_predictions, label='Predicted Housing Supply', linestyle='dashed', color='orange')
plt.xlabel('Year')
plt.ylabel('Housing Supply')
plt.legend()
plt.show()
