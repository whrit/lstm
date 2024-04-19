import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, GRU, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply, BatchNormalization, Flatten, Input
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression   
from scikeras.wrappers import KerasRegressor
import talib
import logging
import datetime
import tensorrt
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def cpugpu():
    import tensorflow as tf
    # Check if GPU is available and print the list of GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found GPU: {gpu}")
    else:
        print("No GPU devices found.")
    
    if gpus:
        try:
            # Specify the GPU device to use (e.g., use the first GPU)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            
            # Test TensorFlow with a simple computation on the GPU
            with tf.device('/GPU:0'):
                a = tf.constant([1.0, 2.0, 3.0])
                b = tf.constant([4.0, 5.0, 6.0])
                c = a * b

            print("GPU is available and TensorFlow is using it.")
            print("Result of the computation on GPU:", c.numpy())
        except RuntimeError as e:
            print("Error while setting up GPU:", e)
    else:
        print("No GPU devices found, TensorFlow will use CPU.")
        
# Setup logging
logging.basicConfig(filename='logfile.log', filemode='a',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting script execution.")

# Check TensorFlow version
logging.info(f"TensorFlow Version: {tf.__version__}")

def check_model_existence(model_path):
    """Check if the model file exists at the specified path."""
    return os.path.exists(model_path)

def user_choice_to_continue(existing_model):
    """Prompt user to decide to use existing model or retrain."""
    response = input(f"Model found at {existing_model}. Do you want to use it? (yes/no): ").strip().lower()
    return response == 'yes'

def load_model(model_path):
    """Load the model from the specified path."""
    return tf.keras.models.load_model(model_path)

# Define the path to the model
model_path = 'trained_model.keras'

# Check if the model exists
if check_model_existence(model_path):
    if user_choice_to_continue(model_path):
        # Load the existing model
        model = load_model(model_path)
        print("Model loaded successfully.")
    else:
        # Proceed with training a new model
        print("Starting new training session...")
        # (Insert your model training code here)
else:
    print("No existing model found. Starting new training session...")
    # (Insert your model training code here)

# Fetch AAPL data
logging.info("Fetching stock_data data...")
stock_data = yf.download('SPY', start='2019-01-01', end='2024-02-16')

# Swap "Adj Close" data into the "Close" column
logging.info("Swapping 'Adj Close' data into 'Close' column...")
stock_data['Close'] = stock_data['Adj Close']

# Remove the "Adj Close" column
logging.info("Removing 'Adj Close' column...")
stock_data = stock_data.drop(columns=['Adj Close'])

# Display the first few rows of the dataframe
logging.info("First few rows of the dataframe:")
logging.info(stock_data.head())

# Checking for missing values
logging.info("Checking for missing values...")
logging.info(stock_data.isnull().sum())

# Filling missing values, if any
logging.info("Filling missing values, if any...")
stock_data.fillna(method='ffill', inplace=True)  # Forward fill to maintain continuity in stock data
stock_data.dropna(inplace=True)  

logging.info("Calculating technical indicators...")
stock_data['SMA_10'] = talib.SMA(stock_data['Close'], timeperiod=10)
stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
stock_data['EMA_20'] = talib.EMA(stock_data['Close'], timeperiod=20)
stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)
stock_data['STOCH_K'], stock_data['STOCH_D'] = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
stock_data['MACD'], stock_data['MACDSIGNAL'], stock_data['MACDHIST'] = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
stock_data['ADX'] = talib.ADX(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['OBV'] = talib.OBV(stock_data['Close'], stock_data['Volume'])
stock_data['ATR'] = talib.ATR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['BBANDS_UPPER'], stock_data['BBANDS_MIDDLE'], stock_data['BBANDS_LOWER'] = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
stock_data['MOM'] = talib.MOM(stock_data['Close'], timeperiod=10)
stock_data['CCI'] = talib.CCI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['WILLR'] = talib.WILLR(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['TSF'] = talib.TSF(stock_data['Close'], timeperiod=14)
stock_data['TRIX'] = talib.TRIX(stock_data['Close'], timeperiod=30)
stock_data['ULTOSC'] = talib.ULTOSC(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
stock_data['ROC'] = talib.ROC(stock_data['Close'], timeperiod=10)
stock_data['PLUS_DI'] = talib.PLUS_DI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['MINUS_DI'] = talib.MINUS_DI(stock_data['High'], stock_data['Low'], stock_data['Close'], timeperiod=14)
stock_data['PLUS_DM'] = talib.PLUS_DM(stock_data['High'], stock_data['Low'], timeperiod=14)
stock_data['MINUS_DM'] = talib.MINUS_DM(stock_data['High'], stock_data['Low'], timeperiod=14)

logging.info("Checking for NaN values after calculating technical indicators...")
logging.info(stock_data.isnull().sum())

logging.info("Forward-filling NaN values...")
stock_data.fillna(method='ffill', inplace=True)  # Forward fill to maintain continuity in stock data
stock_data.dropna(inplace=True)  
# Perform correlation analysis
logging.info("Performing correlation analysis...")
corr_matrix = stock_data[['Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC', 'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM']].corr()
logging.info("Correlation Matrix:\n%s", corr_matrix)

# Select features based on correlation with the target variable
logging.info("Selecting features based on correlation with the target variable...")
k = 10  # Number of top features to select
selector = SelectKBest(score_func=f_regression, k=k)
selected_features = selector.fit_transform(stock_data[['Volume', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC', 'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM']], stock_data['Close'])
selected_feature_names = stock_data[['Volume', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC', 'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM']].columns[selector.get_support()].tolist()
logging.info("Selected Features (Correlation): %s", selected_feature_names)

# Perform recursive feature elimination
logging.info("Performing recursive feature elimination...")
estimator = LinearRegression()
rfe = RFE(estimator, n_features_to_select=k, step=1)
rfe.fit(stock_data[['Volume', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC', 'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM']], stock_data['Close'])
selected_feature_names_rfe = stock_data[['Volume', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI', 'STOCH_K', 'STOCH_D', 'MACD', 'MACDSIGNAL', 'MACDHIST', 'ADX', 'OBV', 'ATR', 'BBANDS_UPPER', 'BBANDS_MIDDLE', 'BBANDS_LOWER', 'MOM', 'CCI', 'WILLR', 'TSF', 'TRIX', 'ULTOSC', 'ROC', 'PLUS_DI', 'MINUS_DI', 'PLUS_DM', 'MINUS_DM']].columns[rfe.support_].tolist()
logging.info("Selected Features (RFE): %s", selected_feature_names_rfe)

# Update the selected features based on the feature selection results
logging.info("Updating selected features based on feature selection results...")
selected_features = list(set(selected_feature_names + selected_feature_names_rfe + ['Close']))
logging.info("Final Selected Features: %s", selected_features)

feature_importances = selector.scores_
feature_importances_rfe = rfe.ranking_
logging.info("Feature Importances (SelectKBest):")
for feature, importance in zip(selected_feature_names, feature_importances):
    logging.info("%s: %s", feature, importance)
logging.info("Feature Importances (RFE):")
for feature, importance in zip(selected_feature_names_rfe, feature_importances_rfe):
    logging.info("%s: %s", feature, importance)

# Define the scaler
logging.info("Defining the scaler...")
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the selected features
logging.info("Scaling the selected features...")
stock_data_scaled = scaler.fit_transform(stock_data[selected_features])

logging.info("Preparing the data for training...")
X = []
y = []

for i in range(60, len(stock_data_scaled)):
    X.append(stock_data_scaled[i-60:i, :])
    y.append(stock_data_scaled[i, 0])

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train, y_train = np.array(X_train), np.array(y_train)

def create_model(lstm_units=50, dropout_rate=0.2, dense_units=32, optimizer='adam', learning_rate=0.001):
    # Input layer
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # LSTM layers with return_sequences=True for both to use in attention
    x = LSTM(units=lstm_units, return_sequences=True)(inputs)
    x = LSTM(units=lstm_units, return_sequences=True)(x)
    
    # Attention mechanism
    attention = AdditiveAttention(name='attention_weight')
    attention_result = attention([x, x])  # Self attention
    x = Multiply()([x, attention_result])
    
    # More dense layers
    x = Dense(units=dense_units, activation='relu')(x)
    x = Dense(units=dense_units//2, activation='relu')(x)
    x = Flatten()(x)  # Flatten before final Dense layer
    output = Dense(1)(x)
    
    # Adding Dropout and Batch Normalization
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)
    
    # Creating and compiling the model
    model = Model(inputs=inputs, outputs=output)
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mean_squared_error')
    
    return model

# Hyperparameter tuning using RandomizedSearchCV
logging.info("Performing hyperparameter tuning...")
param_distributions = {
    'model__lstm_units': [32, 64, 128],
    'model__dense_units': [16, 32, 64],
    'model__dropout_rate': [0.1, 0.2, 0.3],
    'model__optimizer': ['adam', 'rmsprop', 'sgd'],
    'model__learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150]
}

model = KerasRegressor(model=create_model)
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, cv=5, n_iter=10)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
best_params = random_search.best_params_
logging.info("Best parameters: %s", best_params)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_checkpoint = ModelCheckpoint(f'best_model_{timestamp}.keras', save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
tensorboard = TensorBoard(log_dir='./logs')
csv_logger = CSVLogger('training_log.csv')
callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

logging.info("Training the model...")
history = best_model.fit(X_train, y_train, validation_split=0.2, callbacks=callbacks_list)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Evaluate the model on the test data
logging.info("Evaluating the model on test data...")
test_loss = best_model.score(X_test, y_test)
logging.info("Test Loss: %s", test_loss)

# Making predictions
logging.info("Making predictions on test data...")
y_pred = best_model.predict(X_test)

# Calculating evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

logging.info("Mean Absolute Error: %s", mae)
logging.info("Root Mean Square Error: %s", rmse)
logging.info("R-squared: %s", r2)
logging.info("Mean Absolute Percentage Error: %s", mape)

# Saving the trained model and scaler object
logging.info("Saving the trained model and scaler object...")
best_model.model_.save('trained_model.keras')

import joblib
joblib.dump(scaler, 'scaler.pkl')

# Fetch the latest 60 days of AAPL stock data
logging.info("Fetching the latest 60 days of AAPL stock data...")
data = yf.download('SPY', period='60d', interval='1d')
print("Columns after downloading data:", stock_data.columns)
stock_data.head()

# Checking for missing values
logging.info("Checking for missing values...")
logging.info(stock_data.isnull().sum())

logging.info("Checking for missing values...")
if data.isnull().sum().sum() > 0:
    logging.info("NaN values found, handling...")
    data.fillna(method='ffill', inplace=True)  # Forward fill to maintain continuity
    data.dropna(inplace=True)  # Drop remaining NaNs if any

    if data.empty:
        logging.error("Data is empty after filling NaN values. Adjust the filling strategy or check the data source.")

# Calculate the technical indicators for the latest data
logging.info("Calculating technical indicators for the latest data...")
data['SMA_10'] = talib.SMA(data['Close'], timeperiod=10)
data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)
data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
data['STOCH_K'], data['STOCH_D'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
data['MACD'], data['MACDSIGNAL'], data['MACDHIST'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
data['ADX'] = talib.ADX(data['High'], data['Low'], data['Close'], timeperiod=14)
data['OBV'] = talib.OBV(data['Close'], data['Volume'])
data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
data['BBANDS_UPPER'], data['BBANDS_MIDDLE'], data['BBANDS_LOWER'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
data['MOM'] = talib.MOM(data['Close'], timeperiod=10)
data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
data['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
data['TSF'] = talib.TSF(data['Close'], timeperiod=14)
data['TRIX'] = talib.TRIX(data['Close'], timeperiod=30)
data['ULTOSC'] = talib.ULTOSC(data['High'], data['Low'], data['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
data['ROC'] = talib.ROC(data['Close'], timeperiod=10)
data['PLUS_DI'] = talib.PLUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
data['MINUS_DI'] = talib.MINUS_DI(data['High'], data['Low'], data['Close'], timeperiod=14)
data['PLUS_DM'] = talib.PLUS_DM(data['High'], data['Low'], timeperiod=14)
data['MINUS_DM'] = talib.MINUS_DM(data['High'], data['Low'], timeperiod=14)

if np.any(np.isnan(data)):
    logging.error("NaN values detected after processing indicators")
else:
    logging.info("No NaN values present after processing indicators")

logging.info("Forward-filling NaN values...")
data.fillna(method='ffill', inplace=True)  # Forward fill to maintain continuity in stock data
data.dropna(inplace=True)  
    
# Select the same features as used in training
logging.info("Selecting the same features as used in training...")
latest_data = data[selected_features]

if latest_data.empty:
    logging.error("Selected features resulted in an empty DataFrame. Check the selected_features list for errors.")
else:
    # Scaling
    logging.info("Scaling the latest data...")
    scaler = MinMaxScaler()
    scaler.fit(latest_data)  # Use the scaler that was fit on the training data
    scaled_latest_data = scaler.transform(latest_data)

    # Reshape data for prediction
    logging.info("Preparing the input data for prediction...")
    current_batch = scaled_latest_data[-60:].reshape(1, 60, len(selected_features))

# Predict the next 4 days iteratively
logging.info("Predicting the next 4 days iteratively...")
predicted_prices = []

for i in range(4):  # Predicting 4 days
    logging.info(f"Predicting day {i+1}...")

    # Get the prediction (next day)
    next_prediction = best_model.predict(current_batch)
    logging.info(f"Predicted price (scaled): {next_prediction[0][0]}")

    # Reshape the prediction to fit the batch dimension
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)

    # Append the prediction to the batch used for predicting
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)

    # Inverse transform the prediction to the original price scale
    predicted_price = scaler.inverse_transform(next_prediction.reshape(1, -1))[0, 0]
    predicted_prices.append(predicted_price)
    logging.info(f"Predicted price (original scale): {predicted_price}")

logging.info("Predicted Stock Prices for the next 4 days: %s", predicted_prices)