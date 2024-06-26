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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import logging
import talib
import yfinance as yf
import traceback
import warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Fetch SPY data
logging.info("Fetching stock_data data...")
stock_data = yf.download('SPY', start='2004-01-01', end='2024-05-17')

# Swap "Adj Close" data into the "Close" column
logging.info("Swapping 'Adj Close' data into 'Close' column...")
stock_data['Close'] = stock_data['Adj Close']

# Remove the "Adj Close" column
logging.info("Removing 'Adj Close' column...")
stock_data = stock_data.drop(columns=['Adj Close'])

# Checking for missing values
logging.info("Checking for missing values...")
logging.info(stock_data.isnull().sum())

# Filling missing values, if any
logging.info("Filling missing values, if any...")
stock_data.ffill(inplace=True)
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

data_close = stock_data['Close'].values.reshape(-1, 1)
split = int(len(data_close) * 0.9)
data_close_train, data_close_test = data_close[:split], data_close[split:]

# Normalize the Close prices
scaler = MinMaxScaler(feature_range=(-1, 1))
price_data_train = scaler.fit_transform(data_close_train).flatten()
price_data_test = scaler.transform(data_close_test).flatten()

# Variable to control the number of future steps to predict
steps = 1

# Data Preprocessing Function modified for multi-step
def create_sequences(data, sequence_length, steps):
    xs, ys = [], []
    for i in range(len(data) - sequence_length - steps + 1):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length:i + sequence_length + steps]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences with modified function
sequence_length = 1

# Split the data
X_train, y_train = create_sequences(price_data_train, sequence_length, steps)
X_test, y_test = create_sequences(price_data_test, sequence_length, steps)

# Convert to PyTorch tensors, adjusted for multi-step
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader instances, adjusted for multi-step
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# Define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Define the TransformerModel class
class TransformerModel(nn.Module):
    def __init__(self, n_features, d_model, n_heads, n_hidden, n_layers, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, n_layers)
        self.decoder = nn.Linear(d_model, n_features)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = x.unsqueeze(-1)
        if self.src_mask is None or self.src_mask.size(0) != x.size(1):
            mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
            self.src_mask = mask
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, self.src_mask)
        output = self.decoder(output)
        return output.squeeze(-1)[:, -steps:]

# Define the custom scikit-learn estimator
class TransformerRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_layers=4, n_heads=8, n_hidden=512, dropout=0.1, lr=0.001, batch_size=64, epochs=50, patience=5):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model = None

    def fit(self, X, y):
        n_features = steps
        self.model = TransformerModel(n_features=n_features, d_model=128, n_heads=self.n_heads, n_hidden=self.n_hidden, n_layers=self.n_layers, dropout=self.dropout)
        self.model.to(device)

        # Create DataLoader instances
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Loss Function
        criterion = nn.MSELoss()

        # Optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        # Scheduler - OneCycleLR
        scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=self.epochs)

        # Early Stopping
        early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        # GradScaler for mixed precision training
        scaler = GradScaler()

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                sequences, labels = batch
                sequences, labels = sequences.to(device), labels.to(device)
                with autocast():
                    predictions = self.model(sequences)
                    loss = criterion(predictions, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

            validation_loss = self.evaluate(dataloader, criterion)
            print(f'Epoch {epoch+1}: Training Loss: {total_loss/len(dataloader)}, Validation Loss: {validation_loss}')

            early_stopping(validation_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # Load the last checkpoint with the best model
        self.model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

        return self

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                sequences, labels = batch
                sequences, labels = sequences.to(device), labels.to(device)
                predictions = self.model(sequences)
                loss = criterion(predictions, labels)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = self.model(X_tensor)
        return predictions.cpu().numpy()

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_layers': [2, 4, 6],
    'n_heads': [4, 8, 12],
    'n_hidden': [256, 512, 1024],
    'dropout': [0.1, 0.2, 0.3],
    'lr': [0.001, 0.0001],
    'batch_size': [32, 64, 128]
}

# Perform grid search or random search
search = GridSearchCV(estimator=TransformerRegressor(), param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
search.fit(X_train, y_train)

# Get the best model and hyperparameters
best_model = search.best_estimator_
best_params = search.best_params_
print("Best hyperparameters:", best_params)

# Increase model complexity
n_features = steps
complex_model = TransformerModel(n_features=n_features, d_model=128, n_heads=8, n_hidden=1024, n_layers=6, dropout=0.2)

# Move models to appropriate device
best_model.model.to(device)
complex_model.to(device)

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.AdamW(best_model.model.parameters(), lr=best_params['lr'])
optimizer_complex = optim.AdamW(complex_model.parameters(), lr=best_params['lr'])

# Scheduler - OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)
scheduler_complex = OneCycleLR(optimizer_complex, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=50)

warnings.filterwarnings("error", category=UserWarning, module="torch.optim.lr_scheduler")

def train_model(model, train_loader, test_loader, optimizer, criterion, scheduler, epochs, patience):
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)

            with autocast():
                try:
                    predictions = model(sequences)
                    loss = criterion(predictions, labels)
                except Warning as w:
                    print("Warning occurred:")
                    print(str(w))
                    traceback.print_exc()
            
            # Get the scaler state BEFORE updating
            before_step_scale = scaler.get_scale()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()  

            # Check if the scaler skipped the optimizer step
            if before_step_scale == scaler.get_scale():
                scheduler.step()  # Only step the scheduler if the optimizer was updated
            else:
                print("Optimizer step was skipped due to inf/NaN gradients. Skipping scheduler step.")

        # Call optimizer.step() once per epoch, outside of the loop
        optimizer.step()

        validation_loss = evaluate(model, test_loader, criterion)
        print(f'Epoch {epoch+1}: Training Loss: {total_loss/len(train_loader)}, Validation Loss: {validation_loss}')

        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt', map_location=device))

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Run the training loop for the best model
train_model(best_model.model, train_loader, test_loader, optimizer, criterion, scheduler, epochs=50, patience=5)

# Run the training loop for the complex model
train_model(complex_model, train_loader, test_loader, optimizer_complex, criterion, scheduler_complex, epochs=50, patience=5)

# Ensemble modeling
ensemble_models = [
    best_model.model,
    complex_model,
    TransformerModel(n_features=n_features, d_model=256, n_heads=8, n_hidden=512, n_layers=4, dropout=0.1)
]

ensemble_preds = []
for model in ensemble_models:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            sequences, labels = batch
            sequences, labels = sequences.to(device), labels.to(device)
            predictions = model(sequences)
            preds.extend(predictions.view(-1).cpu().numpy())
    ensemble_preds.append(preds) 

ensemble_preds = np.mean(ensemble_preds, axis=0)

# Evaluate on test data, adjusted for multi-step
test_loss = evaluate(best_model.model, test_loader, criterion)
print(f'Test Loss: {test_loss}')

# Calculate and print additional metrics, adjusted for multi-step
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        sequences, labels = batch
        sequences, labels = sequences.to(device), labels.to(device)
        predictions = best_model.model(sequences)
        y_pred.extend(predictions.view(-1).cpu().numpy())
        y_true.extend(labels.view(-1).cpu().numpy())

# Convert predictions and true values to arrays
y_pred = np.array(y_pred).reshape(-1, steps)
y_true = np.array(y_true).reshape(-1, steps)

# Calculate MAE and RMSE for each step and then average (if multi-step)
mae = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
print(f'MAE: {np.mean(mae)}')
print(f'RMSE: {np.mean(rmse)}')

# Evaluate the ensemble predictions
ensemble_preds_reshaped = ensemble_preds.reshape(-1, steps)
mae_ensemble = mean_absolute_error(y_true, ensemble_preds_reshaped, multioutput='raw_values')
rmse_ensemble = np.sqrt(mean_squared_error(y_true, ensemble_preds_reshaped, multioutput='raw_values'))
print(f'Ensemble MAE: {np.mean(mae_ensemble)}')
print(f'Ensemble RMSE: {np.mean(rmse_ensemble)}')

import matplotlib.pyplot as plt

# Calculate indices for x-axis
time_steps = np.arange(len(y_true))

# Plot actual vs predicted prices
plt.figure(figsize=(8, 7))
plt.plot(time_steps, y_true, label='Actual Prices', color='blue', linewidth=2)
plt.plot(time_steps, y_pred, label='Predicted Prices', color='red', linewidth=2)

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
num_train_pairs = calculate_input_output_pairs(num_train_obs, sequence_length)
num_test_pairs = calculate_input_output_pairs(num_test_obs, sequence_length)

print(f'Number of input-output pairs in the training dataset: {num_train_pairs}')
print(f'Number of input-output pairs in the testing dataset: {num_test_pairs}')