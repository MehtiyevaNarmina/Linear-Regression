# Linear-Regression from scratch
import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
n = 1000
area = np.random.rand(n) * 100  # Example feature: area in square meters
price = 3 * area + 7  # Example true relationship: price = 3 * area + 7
noisy_price = price + np.random.randn(n) * 10  # Adding noise to the data

# Hyperparameters
iterations = 1200
learning_rate = 0.00001

# Initialize parameters
w = 0
b = 0

def linear_regression(learning_rate, w, b, x, y, n, iterations):
    for i in range(iterations):
        predicted_price = w * x + b
        J = (1/n) * np.sum((predicted_price - y)**2)
        dw = (2/n) * np.sum((predicted_price - y) * x)
        db = (2/n) * np.sum(predicted_price - y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
    return predicted_price, J, w, b

predicted_price, J, w, b = linear_regression(learning_rate, w, b, area, noisy_price, n, iterations)

print("Loss:", J)
print("Weight:", w)
print("Bias:", b)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(area, noisy_price, color='blue', label='Noisy Data')
plt.plot(area, predicted_price, color='red', label='Fitted Line')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
