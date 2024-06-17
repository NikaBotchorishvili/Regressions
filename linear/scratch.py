import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("./datasets/Housing.csv")

Q1 = data["area"].quantile(0.25)
Q3 = data["area"].quantile(0.75)
IQR = Q3 - Q1
data.dropna(inplace=True)
# Define the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
data = data[(data["area"] >= lower_bound) & (data["area"] <= upper_bound)]


def gradient_descent(data, m_now, b_now, learning_rate):

    m_gradient = 0.0
    b_gradient = 0.0

    DATA_LENGTH = len(data)

    for i in range(DATA_LENGTH):
        x = data.iloc[i].area
        y = data.iloc[i].price

        m_gradient += -(2 / DATA_LENGTH) * x * (y - (m_now * x + b_now))
        b_gradient += -(2 / DATA_LENGTH) * (y - (m_now * x + b_now))
    m = m_now - learning_rate * m_gradient
    b = b_now - learning_rate * b_gradient

    return m, b


m = 0
b = 0
learning_rate = 0.00000002
iterations = 100


for i in range(1, iterations + 1):
    m, b = gradient_descent(data, m, b, learning_rate)
    if i % 100 == 0:
        print(f"Iteration: {i}")


plt.scatter(data["area"], data["price"])
plt.plot(data["area"], m * data["area"] + b, color="red")

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Linear Regression Line")


plt.show()
print(f"Final m: {m:.5f}, Final b: {b:.5f}")
