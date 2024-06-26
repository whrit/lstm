import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import GridSearchCV
import logging
import talib
import yfinance as yf
import time
import atexit
import sys
import os 

# Start tracking time
start_time = time.time()

def exit_handler():
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total runtime: {total_time:.2f} seconds")

atexit.register(exit_handler)

log_file = "logfile.log"
if os.path.exists(log_file):
    os.remove(log_file)

logging.info("Checking if MPS is available...")
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        logging.info("MPS not available because the current PyTorch install was not built with MPS enabled.")
    else:
        logging.info("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
else:
    device = torch.device("mps")
    logging.info(f"Using device: {device}")

# Set random seed for reproducibility
logging.info("Setting random seeds for reproducibility...")
np.random.seed(42)
torch.manual_seed(42)

# Fetch AAPL data
logging.info("Fetching SPY stock data...")
stock_data = yf.download('SPY', start='2004-01-01', end='2024-05-17')

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
stock_data.ffill(inplace=True)  # Forward fill to maintain continuity in stock data
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

logging.info("Reshaping data for training and testing...")
data_close = stock_data['Close'].values.reshape(-1, 1)
split = int(len(data_close) * 0.9)
data_close_train, data_close_test = data_close[:split], data_close[split:]

# Normalize the Close prices
logging.info("Normalizing the Close prices...")
scaler = MinMaxScaler(feature_range=(-1, 1))
price_data_train = scaler.fit_transform(data_close_train).flatten()
price_data_test = scaler.transform(data_close_test).flatten()

# Variable to control the number of future steps to predict
steps = 1

# Data Preprocessing Function modified for multi-step
logging.info("Defining function to create sequences...")
def create_sequences(data, sequence_length, steps):
    logging.info(f"Creating sequences with sequence_length={sequence_length} and steps={steps}...")
    xs, ys = [], []
    for i in range(len(data) - sequence_length - steps + 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length:i + sequence_length + steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences with modified function
sequence_length = 1  # Updated sequence length
logging.info(f"Creating training and testing sequences with sequence_length={sequence_length} and steps={steps}...")
X_train, y_train = create_sequences(price_data_train, sequence_length, steps)
X_test, y_test = create_sequences(price_data_test, sequence_length, steps)

# Convert to PyTorch tensors, adjusted for multi-step
logging.info("Converting data to PyTorch tensors...")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create DataLoader instances, adjusted for multi-step
logging.info("Creating DataLoader instances...")
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the PositionalEncoding class
logging.info("Defining the PositionalEncoding class...")
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        logging.info(f"Initializing PositionalEncoding with d_model={d_model}, dropout={dropout}, max_len={max_len}...")
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        logging.info("Applying positional encoding...")
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Define the TransformerModel class
logging.info("Defining the TransformerModel class...")
class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_hidden, n_layers, dropout):
        logging.info(f"Initializing TransformerModel with n_features={n_features}, d_model={d_model}, n_heads={n_heads}, n_hidden={n_hidden}, n_layers={n_layers}, dropout={dropout}...")
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        
        # Adjust d_model to be divisible by n_heads
        d_model = n_heads * (d_model // n_heads)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.decoder = nn.Linear(d_model, n_features)
        self.init_weights()

        # Initialize src_mask as None
        self.src_mask = None

    def _generate_square_subsequent_mask(self, sz):
        logging.info(f"Generating square subsequent mask of size {sz}...")
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        logging.info("Initializing weights for the TransformerModel...")
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        logging.info("Forward pass of the TransformerModel...")
        sequence_length = x.size(1)
        x = x.unsqueeze(-1)  # Add a feature dimension

        # Ensure src_mask is defined
        if self.src_mask is None or self.src_mask.size(0) != sequence_length:
            self.src_mask = self._generate_square_subsequent_mask(sequence_length).to(x.device)

        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)
        return output.squeeze(-1)  # Remove the feature dimension

# Adjust the TransformerEstimator to fit this change
logging.info("Defining the TransformerEstimator class...")
class TransformerEstimator:
    def __init__(self, n_features, d_model, n_heads, n_hidden, n_layers, dropout):
        logging.info(f"Initializing TransformerEstimator with n_features={n_features}, d_model={d_model}, n_heads={n_heads}, n_hidden={n_hidden}, n_layers={n_layers}, dropout={dropout}...")
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None

    def fit(self, X, y):
        logging.info("Fitting TransformerEstimator model...")
        self.model = TransformerModel(
            n_features=self.n_features,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            dropout=self.dropout
        )
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        self.model.to(device)
        self.model.train()
        for _ in range(5):
            for batch in train_loader:
                optimizer.zero_grad()
                sequences, labels = batch
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = self.model(sequences)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        logging.info("Predicting using TransformerEstimator model...")
        self.model.eval()
        with torch.no_grad():
            X = X.to(device)
            predictions = self.model(X)
        return predictions.cpu().numpy()

    def score(self, X, y):
        logging.info("Scoring TransformerEstimator model...")
        predictions = self.predict(X)
        y_np = y.cpu().numpy().flatten()
        predictions_flat = predictions.flatten()
        min_length = min(len(y_np), len(predictions_flat))
        y_np = y_np[:min_length]
        predictions_flat = predictions_flat[:min_length]
        mse = mean_squared_error(y_np, predictions_flat)
        return -mse

    def get_params(self, deep=True):
        logging.info("Getting parameters of TransformerEstimator model...")
        return {
            'n_features': self.n_features,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_hidden': self.n_hidden,
            'n_layers': self.n_layers,
            'dropout': self.dropout
        }

    def set_params(self, **parameters):
        logging.info("Setting parameters of TransformerEstimator model...")
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Define the parameter grid for hyperparameter tuning
logging.info("Defining parameter grid for hyperparameter tuning...")
param_grid = {
    'n_layers': [2, 4, 6],
    'n_heads': [4, 8, 12],
    'n_hidden': [256, 512, 1024],
    'dropout': [0.1, 0.2, 0.3]
}

n_features = steps
logging.info("Initializing TransformerEstimator for GridSearchCV...")
transformer_estimator = TransformerEstimator(
    n_features=n_features,
    d_model=128,
    n_heads=None,
    n_hidden=None,
    n_layers=None,
    dropout=None
)

# Perform grid search
logging.info("Performing GridSearchCV...")
search = GridSearchCV(estimator=transformer_estimator, param_grid=param_grid, cv=5, verbose=2)
search.fit(X_train_tensor, y_train_tensor)

# Get the best model and hyperparameters
logging.info("Fetching the best model and hyperparameters from GridSearchCV...")
best_model = search.best_estimator_.model
best_params = search.best_params_
logging.info(f"Best hyperparameters: {best_params}")

# Increase model complexity
logging.info("Initializing complex TransformerModel...")
complex_model = TransformerModel(
    n_features=n_features,
    d_model=128,
    n_heads=12,
    n_hidden=1024,
    n_layers=6,
    dropout=0.2
)

# Move models to appropriate device
logging.info("Moving models to device...")
best_model.to(device)
complex_model.to(device)

# Loss Function
logging.info("Defining loss function...")
criterion = nn.MSELoss()

# Optimizer
logging.info("Defining optimizers for models...")
optimizer = optim.AdamW(best_model.parameters(), lr=0.001)
optimizer_complex = optim.AdamW(complex_model.parameters(), lr=0.001)

# Scheduler - OneCycleLR
logging.info("Defining learning rate schedulers for models...")
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)
scheduler_complex = OneCycleLR(optimizer_complex, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)

# Early Stopping
logging.info("Defining EarlyStopping class...")
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        logging.info(f"Initializing EarlyStopping with patience={patience}, delta={delta}, path={path}...")
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        logging.info(f"Calling EarlyStopping with val_loss={val_loss}...")
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        logging.info(f'Saving checkpoint for EarlyStopping with val_loss={val_loss}...')
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)  # Directly save the model's state_dict
        self.val_loss_min = val_loss

best_model.to(device)
complex_model.to(device)

# Training function
logging.info("Defining training function...")
def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, patience):
    logging.info(f"Starting training for {epochs} epochs with patience={patience}...")
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Create a GradScaler instance
    scaler = GradScaler()
    
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}...")
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device) 
            
            # Use autocast context manager
            with autocast(): 
                predictions = model(sequences)
                loss = criterion(predictions, labels)

            # Backpropagation and Optimization with AMP
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        
        validation_loss = evaluate(model, test_loader, criterion)
        logging.info(f'Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader)}, Validation Loss: {validation_loss}')
        
        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break
    
    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

# Evaluation function
logging.info("Defining evaluation function...")
def evaluate(model, val_loader, criterion):
    logging.info("Evaluating model...")
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            sequences, labels = batch
            # Move data to GPU for evaluation
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Run the training loop for the best model
logging.info("Training the best model...")
train_model(best_model, train_loader, test_loader, optimizer, criterion, scheduler, epochs=50, patience=5)

# Run the training loop for the complex model
logging.info("Training the complex model...")
train_model(complex_model, train_loader, test_loader, optimizer_complex, criterion, scheduler_complex, epochs=50, patience=5)

# Ensemble modeling
logging.info("Ensemble modeling with multiple models...")
ensemble_models = [
    best_model,
    complex_model,
    TransformerModel(
        n_features=n_features,
        d_model=256,
        n_heads=8,
        n_hidden=512,
        n_layers=4,
        dropout=0.1
    )
]

ensemble_preds = []
for model in ensemble_models:
    model.to(device)  # Move the model to the appropriate device
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            # Move data to GPU for ensemble prediction
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            preds.extend(predictions.view(-1).cpu().numpy())  
    # Move predictions back to CPU for NumPy operations
    ensemble_preds.append(preds) 

ensemble_preds = np.mean(ensemble_preds, axis=0)
# ensemble_preds = np.average(ensemble_preds, axis=0, weights=[0.4, 0.3, 0.3])  # Example weights

# Evaluate on test data, adjusted for multi-step
logging.info("Evaluating best model on test data...")
test_loss = evaluate(best_model, test_loader, criterion)
logging.info(f'Test Loss: {test_loss}')

# Calculate and print additional metrics, adjusted for multi-step
logging.info("Calculating additional metrics for best model predictions...")
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device)
        predictions = best_model(sequences)
        # Adjust shape for multi-step predictions
        y_pred.extend(predictions.view(-1).cpu().numpy())
        y_true.extend(labels.view(-1).cpu().numpy())

# Convert predictions and true values to arrays
y_pred = np.array(y_pred).reshape(-1, steps)
y_true = np.array(y_true).reshape(-1, steps)

# Calculate MAE and RMSE for each step and then average (if multi-step)
logging.info("Calculating MAE and RMSE for best model predictions...")
mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
logging.info(f'MAE: {np.mean(mae)}')
logging.info(f'RMSE: {np.mean(rmse)}')

# Evaluate the ensemble predictions
logging.info("Evaluating ensemble predictions...")
ensemble_preds_reshaped = ensemble_preds.reshape(-1, steps)
mae_ensemble = mean_absolute_error(y_true, ensemble_preds_reshaped, multioutput='raw_values')
rmse_ensemble = np.sqrt(mean_squared_error(y_true, ensemble_preds_reshaped, multioutput='raw_values'))
logging.info(f'Ensemble MAE: {np.mean(mae_ensemble)}')
logging.info(f'Ensemble RMSE: {np.mean(rmse_ensemble)}')

import matplotlib.pyplot as plt

# Plot actual vs predicted prices
logging.info("Plotting actual vs predicted prices...")
plt.figure(figsize=(8, 7))  # Size of the plot
plt.plot(np.arange(len(y_true)), y_true, label='Actual Prices', color='blue', linewidth=2)  # Actual prices in blue
plt.plot(np.arange(len(y_pred)), y_pred, label='Predicted Prices', color='red', linewidth=2)  # Predicted prices in red

# Add labels and title for clarity
plt.title('SPY Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()

def calculate_input_output_pairs(data_length, sequence_length):
    logging.info(f"Calculating input-output pairs with data_length={data_length} and sequence_length={sequence_length}...")
    # For a given sequence length, the number of input-output pairs is reduced by sequence_length
    # because the last (sequence_length) observations cannot be used to predict a subsequent value.
    num_pairs = data_length - sequence_length
    return num_pairs

# Total number of observations in the dataset
total_observations = 5036

# Split ratio for training data (90% as per your description)
train_split_ratio = 0.9

# Sequence length (1 day as per your description)
sequence_length = 1

# Calculate the split index
split_index = int(total_observations * train_split_ratio)

# Calculate the number of observations in the training and testing datasets
num_train_obs = split_index
num_test_obs = total_observations - split_index

# Calculate and print the number of input-output pairs for both datasets
logging.info("Calculating number of input-output pairs for training and testing datasets...")
num_train_pairs = calculate_input_output_pairs(num_train_obs, sequence_length)
num_test_pairs = calculate_input_output_pairs(num_test_obs, sequence_length)

logging.info(f'Number of input-output pairs in the training dataset: {num_train_pairs}')
logging.info(f'Number of input-output pairs in the testing dataset: {num_test_pairs}')