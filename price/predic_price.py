import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('price/harga.csv')

X = df[['Day']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

day_101 = np.array([[101]])
predicted_price = model.predict(day_101)[0]

predicted_price_rupiah = f"Rp {predicted_price:,.2f}"

print(f"price in day 101 : {predicted_price_rupiah}")
