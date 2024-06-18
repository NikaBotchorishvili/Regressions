import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("D:/Projects/python/ml/regressions/datasets/Housing.csv")

Q1 = data["area"].quantile(0.25)
Q3 = data["area"].quantile(0.75)
IQR = Q3 - Q1
data.dropna(inplace=True)
# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
data = data[(data["area"] >= lower_bound) & (data["area"] <= upper_bound)]


def gradient_descent(data, m_now, b_now, learning_rate, iterations):

    m_gradient = 0.0
    b_gradient = 0.0

    DATA_LENGTH = len(data)
    areas = data["area"].values
    prices = data["price"].values
    for _ in range(iterations):
        y_prediction = m_now * areas + b_now
        error = prices - y_prediction
        
        m_gradient += -(2 / DATA_LENGTH) * np.dot(areas, error)
        b_gradient += -(2 / DATA_LENGTH) * np.sum(error)
        m = m_now - learning_rate * m_gradient
        b = b_now - learning_rate * b_gradient

    return m, b


m = 0
b = 0
learning_rate = 0.000000000002
iterations = 10000


m, b = gradient_descent(
    data=data, m_now=m, b_now=b, learning_rate=learning_rate, iterations=iterations
)


plt.scatter(data["area"], data["price"])
plt.plot(data["area"], m * data["area"] + b, color="red")

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression Line")


plt.show()
print(f"Final m: {m:.5f}, Final b: {b:.5f}")
