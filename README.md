# ARIMA-time-series-forecasting
Time Series Forecasting using Auto Regressive Integrated Moving Average

### Features

- Two models using statsmodels ARIMA module
- A test dataset for the demo model using fixed dates with actuals to compare forecasted values to
- Graphs of the above to visualize accuracy of forecasts

#### Notes:
- This model uses a *week* duration to forecast a company or user's next *week* of output values.
- The demo model uses a fixed date range of:
    - Training Window: 2021.04.19 - 2021.05.17
    - Forecasting Window: 2021.05.17 - 2021.05.23
- While the productio model uses a rolling archetype such that it predicts the next week out each day. (Ideal for Cronjobs)
- All data has been mutated such that it is not representative of any real-world company's value.

### Example Demo Outputs

![Test Company A - Training, Forecast and Output](https://github.com/kanerhee/ARIMA-time-series-forecasting/blob/main/static/companyAgraph.jpg)
![Test Company B - Training, Forecast and Output](https://github.com/kanerhee/ARIMA-time-series-forecasting/blob/main/static/companyBgraph.jpg)
![Test Company C - Training, Forecast and Output](https://github.com/kanerhee/ARIMA-time-series-forecasting/blob/main/static/companyCgraph.jpg)
