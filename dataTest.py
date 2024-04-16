import yfinance as yf
import logging

def fetch_stock_data(symbol, start_date, end_date, output_file):
    logging.info(f"Fetching {symbol} stock data...")
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    logging.info("Swapping 'Adj Close' data into 'Close' column...")
    stock_data['Close'] = stock_data['Adj Close']
    
    logging.info("Removing 'Adj Close' column...")
    stock_data = stock_data.drop(columns=['Adj Close'])
    
    logging.info(f"Saving data to {output_file}...")
    stock_data.to_csv(output_file)
    
    logging.info("Data saved successfully.")
    
    return stock_data

stock_data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01', 'aapl_stock_data.csv')