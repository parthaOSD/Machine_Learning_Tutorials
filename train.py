import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Dataset (Study Hours -> Result Marks)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
y = np.array([35, 40, 50, 55, 65, 70, 80, 90])

# Train Model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("linear_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as linear_model.pkl")