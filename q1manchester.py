import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error


data = pd.read_csv('manchesterdata.csv')
data = data[['Year', 'Total Housing Units']]
data = data.dropna()
data = data[data['Year'] != 2021]
print(data)


data['Year'] = pd.to_datetime(data['Year'], format='%Y')

data.set_index('Year', inplace=True)

model = ExponentialSmoothing(data['Total Housing Units'], trend='add', seasonal=None)

future_years = pd.date_range(start=data.index[0] + pd.DateOffset(years=1), periods=82, freq='YE')  # Adjust as needed

best_mse = 1000000000
best_alpha = None
best_beta = None

for a in range(1, 10):
    for b in range(1, 10):
        alpha = a/10
        beta = b/10
        fit_model = model.fit(smoothing_level=alpha, smoothing_trend=beta)
        future_predictions = fit_model.predict(start=0, end=len(future_years) - 1)
        mse = mean_squared_error(data['Total Housing Units'], future_predictions[:28])
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_beta = beta

print(best_alpha, best_beta)
fit_model = model.fit(smoothing_level=best_alpha, smoothing_trend=best_beta)
future_predictions = fit_model.predict(start=0, end=len(future_years) - 1)

for year in [29, 41, 51, 81]:
    print(f'Year {year}: Predicted Housing Supply - {int(future_predictions[year-1])}')

plt.plot(data.index, data['Total Housing Units'], label='Actual Housing Supply')
plt.plot(future_years, future_predictions, label='Predicted Housing Supply', linestyle='dashed', color='orange')
plt.xlabel('Year')
plt.ylabel('Housing Supply')
plt.title('Housing Supply Prediction in Manchester with Double Exponential Smoothing')
plt.legend()
plt.show()
