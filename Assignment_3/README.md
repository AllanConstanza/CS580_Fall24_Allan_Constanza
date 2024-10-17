# CS580_Fall24_Allan_Constanza
We first import the libraries. Pandas is a tool used a lot for data analysis and matplotlib.pyplot is use for plotting graphs. 
We then load the data using read_csv(). The independent and dependent variables are then extracted from the dataset.  
We compute the mean, covariance, and variance using .mean(), data.cov(), independent_variable.var(). After we've calculated the slope and 
intercept using the covariance and variance, we plot the data points and regression line. We use plt.scatter() for the scatter plot and
plt.plot() for the regression line. Finally, we add the labels and plt.show() to display the plot. 