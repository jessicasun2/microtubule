import numpy as np

def ecdf_vals(data):
    '''Return the ECDF values for values of x in a given data in the form of an array'''
    # Find total length of the data
    n = len(data)
    
    # Initialize an array to store the x and y values we get from the data
    x_y_values = np.zeros((n, 2))
    
    # loop through the data and store the value as an x value
    # find the fraction of data points that are less than or equal and add it to the array
    for i in range(n):
        x_y_values[i, 0] = data[i]
        y = (len(data[(data <= data[i])]))/n
        x_y_values[i, 1] = y
        
    return x_y_values

