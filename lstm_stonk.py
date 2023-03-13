'''
This is still a work in progress using yahoo finance with sklearn in LSTM layers
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf
import keras.layers as layers

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM



# Load DAta
company = 'FSCSX'

# Set start and end time to get data from
start = dt.datetime(1985,8,5) # This is the start of their historical data for FSCSX
end = dt.datetime(2023,3,12)

# use yf.download to get the historical data for the ticker and save it as a csv
df = yf.download(company, start, end)
df = df.to_csv("Data/Train/FSCSX_Train.csv")

# assign stock csv to df var
df = pd.read_csv("Data/Train/FSCSX_Train.csv")
df = df[['Date', 'Close']]


# Date in CSV is currently str - Define function to move str to datetime
def str_to_datetime(s):
    # split the s string utilizing the '-'
    split = s.split('-')
    # assign year month and day to the split values
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    # return the year, month, and day as datetime
    return dt.datetime(year=year, month=month, day=day)

# Pass the full date column to str_to_datetime
df['Date'] = df['Date'].apply(str_to_datetime)

# remove the index from the data only using the date
df.index = df.pop('Date')

def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date  = str_to_datetime(last_date_str)

    target_date = first_date
  
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)
    
        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date+dt.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = dt.datetime(day=int(day), month=int(month), year=int(year))
    
        if last_time:
            break
    
        target_date = next_date

        if target_date == last_date:
            last_time = True
    
        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
  
        X = np.array(X)
        for i in range(0, n):
            X[:, i]
            ret_df[f'Target-{n-i}'] = X[:, i]
  
        ret_df['Target'] = Y

        return ret_df

# Start day second time around: '2021-03-25'
windowed_df = df_to_windowed_df(df, '1985-08-08', '2022-01-01', n=3)
print(windowed_df)
input('press enter')

def windowed_df_to_date_X_y(windowed_dataframe):
    # Converst windowed_dataframe to array
    df_as_np = windowed_dataframe.to_numpy()


    dates = df_as_np[:, 0]

    middle_matrix = df_as_np[:, 1:-1]
    X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

    Y = df_as_np[:, -1]

    return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

plt.plot(dates_train, y_train)
plt.plot(dates_val, y_val)
plt.plot(dates_test, y_test)

plt.legend(['Train', 'Validation', 'Test'])
plt.show()

model = Sequential([
layers.Input((3, 1)),
layers.LSTM(64),
layers.Dense(32, activation='relu'),
layers.Dense(32, activation='relu'),
layers.Dense(1)
])

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)


train_predictions = model.predict(X_train).flatten()

plt.plot(dates_train, train_predictions)
plt.plot(dates_train, y_train)
plt.legend(['Training Predictions', 'Training Observations'])
plt.show()