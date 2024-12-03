import random
import requests
from io import StringIO
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class User:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.id = random.randint(1000000000, 9999999999)
        self.stocks = []

    def register(self, username, password):
        self.username = username
        self.password = password
        self.id = random.randint(1000000000, 9999999999)

    def login(self, username, password):
        #check the db
        return username == self.username and password == self.password

    def add_stock(self, stock_name, stock_ticker):
        stock = Stock(stock_name, stock_ticker)
        self.stocks.append(stock)

    def load_user_stocks(self):
        return self.stocks

class Stock:
    def __init__(self, name, ticker):
        self.name = name
        self.ticker = ticker

    def get_data(self):
        key = #API KEY HERE
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.ticker}&apikey={key}&datatype=csv'
        response = requests.get(url)

        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for {self.ticker}: {response.status_code}")
        
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        if 'close' not in df.columns:
            df.to_csv('failed_data.csv')
        return df


class Visualization(ABC):
    @abstractmethod
    def display(self):
        pass
    
    @abstractmethod
    def load_data(self, data):
        pass

class HistoricalDataV(Visualization):
    def __init__(self, name):
        self.name = name
        self.data = None  
    
    def load_data(self, data):
        self.data = data
       
    
    
    def display(self):
        if self.kind == 'volume':
            self.display_volume()
        else:
            self.display_prices()
    
    def display_volume(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call `load_data` first.")
        plt.plot(self.data['volume'])
        plt.title('Historical Volume')
        plt.savefig('./static/visualizations/HistoricalVolume.png')
    
    def display_prices(self):
        if self.data is None:
            raise ValueError("Data not loaded. Please call `load_data` first.")
        plt.plot(self.data['open'])
        plt.title(f'Historical closing Price')
        plt.savefig('./static/visualizations/HistoricalPrices.png')

class PriceForecast(Visualization):
    def __init__(self, name):
        self.name = name
        self.data = None
        self.model = None
    
    def set_model(self, model):
        self.model = model
    
    def load_data(self, data):
        self.data = data
    
    def build_forecast_window(self):
        """Builds the last 60-window for forecasting."""
        if self.data is None:
            raise ValueError("Data not loaded. Please call `load_data` first.")
        if len(self.data) < 60:
            raise ValueError("Insufficient data for a 60-step forecast.")
        
        final_60 = self.data.iloc[-60:]
        window = [[x] for x in final_60['close'].tolist()]
        return [window]

    def recursive_forecasting_days(self, initial_window):
        """Recursive forecast loop."""
        if self.model is None:
            raise ValueError("Model not set. Use `set_model` to set a model.")
        
        num_days = int(input('Enter the number of forecast days as an integer: '))
        preds = []
        current_window = initial_window
        
        for _ in range(num_days):
            current_tensor = torch.tensor(current_window).unsqueeze(0).float()
            with torch.no_grad():
                pred = self.model(current_tensor)
            current_window.pop(0)
            current_window.append([pred.item()])
            preds.append(pred.item())
        
        return preds

    def plot_forecast(self, preds, actuals, model_name):
        """Plots forecast vs actual values."""
        plt.plot(preds, label='Forecast', color='green', linestyle='-', marker='o')
        plt.plot(actuals, label='Actuals', color='red', linestyle='--', marker='x')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.title(f'{model_name} Forecast')
        plt.legend()
        plt.savefig("./static/visualizations/PriceForecast.png")

class GRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, output_size=1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        gru_out, _ = self.gru(x, h0)
        out = self.fc(gru_out[:, -1, :])
        return out
