from flask import Flask, request, jsonify, render_template, redirect, url_for
from models import User, Stock, HistoricalDataV, PriceForecast, GRU
import os

app = Flask(__name__)

# Mock in-memory storage for users
users = {}

# Stock Data Visualization directory
VISUALIZATION_DIR = "static/visualizations"
if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Create and store user
        if username in users:
            return jsonify({'status': 'error', 'message': 'User already exists'})
        user = User(username, password)
        users[username] = user
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Authenticate user
        user = users.get(username)
        if user and user.login(username, password):
            return redirect(url_for('dashboard', username=username))
        return jsonify({'status': 'error', 'message': 'Invalid credentials'})
    return render_template('login.html')

@app.route('/dashboard/<username>', methods=['GET', 'POST'])
def dashboard(username):
    user = users.get(username)
    if not user:
        return redirect(url_for('login'))

    if request.method == 'POST':
        stock_name = request.form['stock_name']
        stock_ticker = request.form['stock_ticker']

        user.add_stock(stock_name, stock_ticker)
        return redirect(url_for('dashboard', username=username))

    stocks = user.load_user_stocks()
    
    # Visualization and data processing for each stock in the dashboard
    stock_visualizations = {}
    for stock in stocks:
        stock_data = Stock(name=stock.name, ticker=stock.ticker).get_data()
        
        # Historical Data Visualization
        historical_data_viz = HistoricalDataV(name=stock.name)
        historical_data_viz.load_data(stock_data)
        historical_data_viz.display_prices()  # Save the plot to the static folder
        historical_data_viz.display_volume()  # Save the volume plot
        
        # Store the saved plot paths in stock_visualizations
        stock_visualizations[stock.name] = {
            'price_plot': f"{VISUALIZATION_DIR}/{stock.name}_HistoricalPrices.png",
            'volume_plot': f"{VISUALIZATION_DIR}/{stock.name}_HistoricalVolume.png"
        }

        stock_visualizations[stock.name] = {
    'price_plot': f"static/visualizations/{stock.name}_HistoricalPrices.png",
    'volume_plot': f"static/visualizations/{stock.name}_HistoricalVolume.png"
}

    
    return render_template('dashboard.html', stocks=stocks, visualizations=stock_visualizations)

@app.route('/stock/data/<ticker>', methods=['GET'])
def stock_data(ticker):
    stock = Stock(name="", ticker=ticker)  # Mock stock object
    data = stock.get_data()
    return jsonify(data.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
