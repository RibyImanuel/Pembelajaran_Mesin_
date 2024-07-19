import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Load data
data = pd.read_csv("LSTM/BTC-USD.csv")
data["Date"] = pd.to_datetime(data["Date"], format="%Y-%m-%d")
data.set_index("Date", inplace=True)
data = data.sort_index()

# Visualize data (chart harga Bitcoin)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Harga Bitcoin', color='blue')
plt.title('Harga Bitcoin Seiring Waktu')
plt.xlabel('Tanggal')
plt.ylabel('Harga (USDT)')
plt.legend()
plt.show()

# Prepare data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

# Define window size
window_size = 60

# Function to create dataset
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

# Create dataset
X, y = create_dataset(scaled_data, window_size)

# Reshape data for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
train_size = int(len(X) * 0.7)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate model
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# # Visualize predictions (chart harga Bitcoin dan prediksinya)
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Harga Bitcoin', color='blue')
plt.plot(data.index[train_size+window_size:], predictions, color='green', label='Prediksi Harga Bitcoin')
plt.title('Prediksi Harga Bitcoin')
plt.xlabel('Tanggal')
plt.ylabel('Harga (USD)')
plt.legend()
plt.show()

# # Memprediksi harga untuk empat tahun ke depan
# future_predictions = []
# last_window = scaled_data[-window_size:]
# last_window = np.reshape(last_window, (1, window_size, 1))

# for i in range(1460):  # Mengubah iterasi menjadi 1460 untuk empat tahun
#     pred = model.predict(last_window)
#     future_predictions.append(pred[0][0])
#     last_window = np.roll(last_window, -1, axis=1)  # Geser nilai ke kiri
#     last_window[0][-1] = pred  # Ganti nilai terakhir dengan prediksi

# # Mengembalikan prediksi ke dalam rentang harga aslinya
# future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# # Menghasilkan tanggal untuk empat tahun ke depan
# last_date = data.index[-1]
# future_dates = pd.date_range(start=last_date, periods=1461)[1:]  # Mengubah jumlah periode menjadi 1461 untuk empat tahun

# # Visualisasi prediksi untuk empat tahun ke depan
# plt.figure(figsize=(16, 10))
# plt.plot(data.index, data['Close'], label='Harga Bitcoin', color='blue')
# plt.plot(future_dates, future_predictions, color='red', label='Prediksi Harga Bitcoin (4 tahun ke depan)')
# plt.title('Prediksi Harga Bitcoin (4 Tahun ke Depan)')
# plt.xlabel('Tanggal')
# plt.ylabel('Harga (USD)')
# plt.legend()
# plt.show()
