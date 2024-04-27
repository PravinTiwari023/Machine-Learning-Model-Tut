# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data: hours studied and corresponding test scores
hours_studied = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # X values (reshaped for sklearn)
test_scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95])  # Y values

# Creating a linear regression model
model = LinearRegression()

# Training the model with our data
model.fit(hours_studied, test_scores)

# Making predictions
hours_to_predict = np.array([5]).reshape(-1, 1)  # Predicting for 5 hours of study
predicted_score = model.predict(hours_to_predict)

# Plotting the data and the fitted line
plt.figure(figsize=(8, 6))
plt.scatter(hours_studied, test_scores, color='blue', label='Actual Data')
plt.plot(hours_studied, model.predict(hours_studied), color='red', label='Fitted Line')
plt.scatter(hours_to_predict, predicted_score, color='green', label='Predicted Value (5 hours)')
plt.xlabel('Hours Studied')
plt.ylabel('Test Score')
plt.title('Linear Regression: Hours Studied vs. Test Score')
plt.legend()
plt.grid(True)
plt.show()

# Output the predicted score
print(f"Predicted test score for 5 hours of study: {predicted_score[0]}")
