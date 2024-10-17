import pandas as pd #Powerful library for data analysis in Python, more convenient way to read and manipulate CSV data
import matplotlib.pyplot as plt #plotting library in Python used for creating high-quality visualizations

# our dataset has no headers
data = pd.read_csv('linear_regression_data.csv', header=None)

# Assigning column names b/c our file doesn't have headers
#first column called X, second called Y
data.columns = ['X', 'Y']


#Our file doesnt have headers, manually assigning names 
X = data['X']  # Independent variable
Y = data['Y']  # Dependent variable


#getting mean of X and Y columns
mean_x = X.mean() 
mean_y = Y.mean()

#getting mean of X and Y columns using data.cov() function
covariance = data.cov().loc['X', 'Y']
variance = X.var()

# Calculate slope (m) and intercept (c)
m = covariance / variance
c = mean_y - m * mean_x

# Plot the data
plt.scatter(X, Y, color = 'green',label='Data Points')

#regression line, y=mx+c
Y_pred = m * X + c
plt.plot(X, Y_pred, color='red', label='Regression Line')

plt.xlabel('X')  # X-axis label
plt.ylabel('Y')  # Y-axis label
plt.title('Linear Regression, Problem 1')
plt.legend()
plt.show()
