'''
This is still a work in progress using yahoo finance with sklearn in LSTM layers
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM



# Load DAta
company = 'FSCSX'

# Set start and end time to get data from
start = dt.datetime(1985,7,28) # This is the start of their historical data for FSCSX
end = dt.datetime(2022,1,1)

# use yf.download to get the historical data for the ticker and save it as a csv
train_data = yf.download(company, start, end)
data = train_data.to_csv("Data/Train/FSCSX_Train.csv")

# assign stock csv to train_data var
train_data = pd.read_csv("Data/Train/FSCSX_Train.csv")
train_data = train_data[['Date', 'Close']]


# Date in CSV is currently str - Define function to move str to datetime
def str_to_datetime(s):
    # split the s string utilizing the '-'
    split = s.split('-')
    # assign year month and day to the split values
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    # return the year, month, and day as datetime
    return dt.datetime(year=year, month=month, day=day)

# Pass the full date column to str_to_datetime
train_data['Date'] = data['Date'].apply(str_to_datetime)



prediction_days = 30

x_train = []
y_train = []

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)
#model.save('predictions.hdf5')

# Test the model on existing data
# Load the test data
test_start = dt.datetime(2022,1,1)
test_end = dt.datetime.now()

test_data = yf.download(company, test_start, test_end)
test_data = test_data.to_csv("Data/Predict/FSCSX_Predict.csv")

test_data = pd.read_csv("Data/Predict/FSCSX_Predict.csv")
test_data = test_data[['Date', 'Close']]

test_data['Date'] = test_data['Date'].apply(str_to_datetime)

actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make preditctions

x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = predicted_prices.reshape(predicted_prices.shape[0], predicted_prices.shape[1])
predicted_prices = scaler.inverse_transform(predicted_prices)
predicted_prices = predicted_prices[-len(test_data):, 0]
print(predicted_prices)

print(predicted_prices.shape)



# Plot Test Predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f"{company} Share Price")
plt.legend()
plt.show()