def arima(dframe, comp_code, day_steps, p,d,q, plotthem):

    def make_backbone():

        current = datetime.date.today() #.strftime('%Y-%m-%d')
        yesterday = current + timedelta(days=-1)
#         yesterdayminus30 = yesterday + timedelta(days=-30)
#         forecastwindow0 = current + timedelta(days = -7)
        forecastwindow1 = current + timedelta(days = -1)
        trainingwindow0 = current + timedelta(days = -35)
#         trainingwindow1 = trainingwindow0 + timedelta(days = 28)
        datelist = [trainingwindow0]
        temp = trainingwindow0 + timedelta(days=1)
        while temp != current:
            datelist.append(temp)
            temp = temp + timedelta(days=1)
        backbone = pd.DataFrame({'activity_date':datelist})
        return backbone

    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return np.array(diff)

    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    backbone = make_backbone()
    dft = dframe.copy()[dframe.copy()['company'] == comp_code]
    dft = dft.groupby(['activity_date']).agg({'pages':'sum'}).reset_index()

    backbone['activity_date'] = pd.to_datetime(backbone['activity_date']).dt.normalize()
    dft['activity_date'] = pd.to_datetime(dft['activity_date']).dt.normalize()
    dft2 = pd.merge(backbone, dft, how='left', on='activity_date').fillna(0.0)
    dft2['activity_date'] = dft2['activity_date'].astype(str)
    dft2 = dft2.set_index('activity_date')

    # For line plot of dataset
    plt.rcParams["figure.figsize"] = (25,10)

    series = dft2.copy()
    split_point = len(series) - day_steps
    dataset, validation = series[0:split_point], series[split_point:]
    dataset.to_csv('dataset.csv', index=False)
    validation.to_csv('validation.csv', index=False)
#     series.plot()
#     plt.show()

    series = pd.read_csv('dataset.csv', header=0)
    X = series.values
    days_in_year = len(series)
    differenced = difference(X, days_in_year)
    model = ARIMA(differenced, order=(d,p,q))
    model_fit = model.fit()

    start_index = len(differenced)
    end_index = start_index + (day_steps - 1)
    forecast = model_fit.predict(start = start_index, end = end_index)

    history = [x for x in X]
    day = 1
    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        history.append(inverted)
        day += 1

    mi = history[-len(validation):]
    m = 0
    hold = []
    while m < len(mi):
        hold.append(mi[m].tolist())
        m += 1
    hold = flatten_list(hold)

    vi = validation.copy().reset_index()
    datelist = list(vi['activity_date'])
    mii = pd.DataFrame({'activity_date':datelist, 'pages':hold})
    mii = mii.set_index('activity_date')
    forecast2 = mii.copy()
    dataset2 = dataset.copy()
    validation2 = validation.copy()
    forecast2['Type'] = 'Forecast'
    dataset2['Type'] = 'Actual'
    validation2['Type'] = 'Actual - Validation'
    dfm = pd.concat([dataset2, validation2, forecast2])
    dfm['pages'] = dfm['pages'].round(2)

    dfm2 = dfm.copy()
    dfm2['company'] = comp_code

    if plotthem:
        fig = dfm2.pivot(columns="Type", values="pages").plot()
        plt.show()
        return dfm2, fig

    return dfm2

def timer(starttime, endtime):
    seconds = 0
    seconds = endtime-starttime
    minutes = int(seconds//60)
    remainder = int(seconds%60)
    print('Process took:', minutes, ' min ', remainder, ' seconds to perform.\n')
    return

############################################################################################################

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import time
import os
import warnings
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


warnings.filterwarnings('ignore')

start = time.time()

# Raw Dataset of Test Data
df = pd.read_csv('arima_test_data.csv')


# These are our FIXED date ranges for training our model and forecasting the next week,
# with actuals to compare our forecast against
trainingwindow0 = datetime.date(2021, 4, 19).strftime('%Y-%m-%d')
trainingwindow1 = datetime.date(2021, 5, 17).strftime('%Y-%m-%d')
forecastwindow0 = datetime.date(2021, 5, 17).strftime('%Y-%m-%d')
forecastwindow1 = datetime.date(2021, 5, 23).strftime('%Y-%m-%d')
print('training window: ', trainingwindow0, ' to ', trainingwindow1)
print('forecast window: ', forecastwindow0, ' to ', forecastwindow1)

# Truncated Dataset of Test Data
df2 = df.copy()
df2 = df2[df2['activity_date'] >= trainingwindow0]
df2 = df2[df2['activity_date'] <= forecastwindow1]

# There are 3 companies in the test dataset. Feel free to choose any company, or loop through all as I did below
companylist = ['Company A', 'Company B', 'Company C']

# Tuning for the ARIMA Model
days_forecasted = 7
p = 7
d = 1
q = 1

outputmaster = []
for company in companylist:
    print(company)
    outputdf, forecastactualplot = arima(df2, company, days_forecasted, p,d,q, True)
    outputmaster.append(outputdf)

end = time.time()

os.remove('dataset.csv')
os.remove('validation.csv')

timer(start, end)

print(outputmaster)
