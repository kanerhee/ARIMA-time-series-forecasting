def timer(starttime, endtime):
    seconds = 0
    seconds = endtime-starttime
    minutes = int(seconds//60)
    remainder = int(seconds%60)
    print('Process took:', minutes, ' min ', remainder, ' seconds to perform.\n')
    return

def open_pickle(filename):
    import pickle
    with open(filename, 'rb') as pickle_file:
        categorical = pickle.load(pickle_file)
        return categorical

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

    # line plot of dataset
    plt.rcParams["figure.figsize"] = (25,10)

    series = dft2.copy()
    split_point = len(series) - day_steps

    dataset, validation = series[0:split_point], series[split_point:]
    dataset.to_csv('dataset.csv', index=False)
    validation.to_csv('validation.csv', index=False)
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

######################################################################################################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import datetime
from datetime import timedelta
import time
import os
from ftplib import FTP
import warnings
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from IPython.display import display_html


warnings.filterwarnings('ignore')

# ****************************************************************** #
# *******************  LOAD YOUR DATA HERE  ************************ #
# ****************************************************************** #
# i.e.
# df = pd.read_pickle('arima_forecast_data.csv')
print('length of df: ', len(df))
end = time.time()
timer(start, end)

current = datetime.date.today()
currentminus30 = current + timedelta(days = -30)
yesterday = (current + timedelta(days = -1)).strftime('%Y-%m-%d')
yesterdayminus7 = (current + timedelta(days= -8)).strftime('%Y-%m-%d')
yesterdayminus30 = (currentminus30 + timedelta(days = -1)).strftime('%Y-%m-%d')
print('current: ', current, '\nyesterday: ', yesterday, '\nyesterdayminus30: ', yesterdayminus30)

forecastwindow0 = current + timedelta(days = -7)
forecastwindow1 = current + timedelta(days = -1)
trainingwindow0 = current + timedelta(days = -35)
trainingwindow1 = trainingwindow0 + timedelta(days = 28)
print('training window: ', trainingwindow0, ' to ', trainingwindow1)
print('forecast window: ', forecastwindow0, ' to ', forecastwindow1)

trainingwindow0 = trainingwindow0.strftime('%Y-%m-%d')
forecastwindow1 = forecastwindow1.strftime('%Y-%m-%d')

######################################################################

df2 = df.copy()
df2 = df2[df2['activity_date'] >= trainingwindow0]
df2 = df2[df2['activity_date'] <= forecastwindow1]
print('length of df truncated to above dates: ', len(df2))

compcounts = pd.DataFrame(df2['company'].value_counts().reset_index())[0:1500]
compcounts.rename(columns={'index':'company', 'company':'frequency'}, inplace=True)
company_list = list(set(compcounts['company']))
print('length of company list: ', len(company_list))

output_list = []
counter = 0
for company in company_list[0:100]:
#     print(counter, ': ', company, '   ', end='')
    print(counter, end='')
    outputdf = arima(df2, company, 7, 7,1,1, False)
    output_list.append(outputdf)
    counter += 1

testmaster = pd.concat(output_list).reset_index()
testmaster.replace({'Actual':'Actual - Training'}, inplace=True)
testmaster.to_csv('testmaster_v5.csv', encoding='utf-8')

end1 = time.time()
timer(start1, end1)

######################################################################

output_list2 = []
plot_list2 = []
counter = 0
for company in company_list[0:100]:
    print(counter, ': ', company, '   ', end='')
#     print(counter, end='')
    outputdf2, plott2 = arima(df2, company, 7, 7,1,1, True)
    output_list2.append(outputdf2)
    plot_list2.append(plott2)
    counter += 1

print('Done. \n\n')

testmaster.head(10)
