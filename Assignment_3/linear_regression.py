import pandas as pd #library for data analysis in Python, more convenient way to read and manipulate CSV data
import matplotlib.pyplot as plt #plotting library in Python used for creating visualizations

# our dataset has no headers
data = pd.read_csv('linear_regression_data.csv', header=None)

# Assigning column names b/c our file doesn't have headers
data.columns = ['independent_variable', 'dependent_variable']


#Our file doesnt have headers, manually assigning names 
independent_variable = data['independent_variable']  # Independent variable
dependent_variable = data['dependent_variable']  # Dependent variable


#getting mean of both columns columns
mean_independent = independent_variable.mean() 
mean_dependent = dependent_variable.mean()

#getting covariance of both columns using data.cov() function
covariance = data.cov().loc['independent_variable', 'dependent_variable']
variance = independent_variable.var()

# Calculate slope (m) and intercept (c)
m = covariance / variance
c = mean_dependent - m * mean_independent

# Plot the data
plt.scatter(independent_variable, dependent_variable, color = 'green',label='Data Points')

#regression line, y=mx+c
dependent_line = m * independent_variable + c


plt.plot(independent_variable, dependent_line, color='red', label='Regression Line')

plt.xlabel('Independent Variable')  # X-axis label
plt.ylabel('Dependent Variable')  # Y-axis label
plt.title('Linear Regression, Problem 1')
plt.legend()
plt.show()
