"""Linear and polynomial regression examples."""

import matplotlib.pyplot as plt
from sklearn import linear_model, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


height = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
weight = [8, 10, 12, 14, 16, 18, 20]

plt.scatter(height, weight, color="black")
plt.xlabel("height")
plt.ylabel("weight")

reg = linear_model.LinearRegression()
reg.fit(height, weight)

x_height = [[12.0]]
print(reg.predict(x_height))


# Train/test split example
x = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y = [8, 10, 12, 14, 16, 18, 20]

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.3, random_state=7
)

print("Training Features", x_train)
print("Training Labels", y_train)
print("Testing Features", x_test)
print("Testing Labels", y_test)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

# Accuracy on test set
result = reg.score(x_test, y_test)
print(f"Accuracy - test set: {result * 100.0:.2f}%")

print(reg.predict(x_test))


# Polynomial regression example
x = [[4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]
y = [16, 25, 36, 49, 64, 81, 100]

lin_reg = LinearRegression()
lin_reg.fit(x, y)

print(lin_reg.predict([[11]]))

polynomial_regression = make_pipeline(
    PolynomialFeatures(degree=1, include_bias=False),
    LinearRegression(),
)

polynomial_regression.fit(x, y)

x_height = [[20.0]]
target_predicted = polynomial_regression.predict(x_height)
print(target_predicted)
