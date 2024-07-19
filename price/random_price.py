import pandas as pd
import numpy as np

# Generate random prices for 100 days
np.random.seed(0)
data = {
    'Day': np.arange(1, 101),
    'Price': np.round(np.random.rand(100) * 100000, 2)  # Random prices between 0 and 100000
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('price/harga.csv', index=False)
