# Stock Price Prediction using LSTM

A Long Short-Term Memory (LSTM) neural network has been implemented to predict stock prices. The LSTM is trained on historical data of a company and is used to predict future stock prices.

## Requirements

- Python 3.x
- numpy
- matplotlib
- pandas
- pandas_datareader
- datetime
- yfinance
- sklearn
- keras

## Installation

1. Clone the repository:
    https://github.com/Orthrus88/stock_predictions

2. Install the required packages:
    pip install -r requirements.txt

3. To show the matplotlib graphs you need to ensure tkinter is installed
    sudo apt isntall python3-tk


## Usage

1. Open the `main.py` file and set the `company`, `start`, and `end` dates for the historical data.

2. Run the `main.py` file. The LSTM model will be trained on the historical data and saved as `predictions.hdf5`.

3. Open the `test.py` file and set the `company`, `test_start`, and `test_end` dates for the test data.

4. Run the `test.py` file. The LSTM model will use the trained weights to predict future stock prices and the actual prices will be plotted against the predicted prices.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.