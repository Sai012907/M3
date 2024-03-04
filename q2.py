import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 4.5, 5.5])

# Bayesian linear regression model
with pm.Model() as model:
    # Priors for the parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)

    # Likelihood function
    mu = alpha + beta * x
    sigma = pm.HalfNormal('sigma', sd=1)
    y_obs = pm.Normal('y_obs', mu=mu, sd=sigma, observed=y)

    # Sample from the posterior distribution
    trace = pm.sample(1000, tune=1000)

# Plotting the results
pm.traceplot(trace)
plt.show()

# Display summary statistics of the posterior distribution
print(pm.summary(trace))

# Predictions for new data
new_x = np.array([6, 7, 8])
with model:
    y_pred = pm.sample_posterior_predictive(trace, samples=500, var_names=['y_obs'], input_vals={'x': new_x})

# Plotting the predictions
plt.scatter(x, y, label='Observed Data')
plt.plot(new_x, np.median(y_pred['y_obs'], axis=0), label='Median Prediction', color='red')
plt.fill_between(new_x, np.percentile(y_pred['y_obs'], 2.5, axis=0), np.percentile(y_pred['y_obs'], 97.5, axis=0), color='red', alpha=0.3, label='95% Credible Interval')
plt.legend()
plt.show()
