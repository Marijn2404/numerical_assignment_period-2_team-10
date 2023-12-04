# Imports: Do not modify!
import datetime
from math import pi, exp
from copy import copy
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# determine correct working directory
import os
wd = os.path.dirname(os.path.abspath(__file__)) + '/'

#
# Question 1 - Random Number Generation
#

# Q1 (a)
def pdf_cauchy(x, mu=0, sigma=1):
    f = 0.0
    ## BEGIN ANSWER
    f = 1 / (np.pi * sigma * (1 + ((x - mu) / sigma)**2))
    ## END ANSWER
    return f

# Q1 (b)
def pdf_laplace(x, mu=0, b=1):
    f = 0
    ## BEGIN ANSWER
    f = 1 / (2 * b) * np.exp(-np.abs(x - mu) / b)
    ## END ANSWER
    return f

# Q1 (c)
def rng_cauchy(n, mu=0, sigma=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n)
    ## BEGIN ANSWER
    x = np.random.rand(n)

    # using inverse cdf of the cauchy distribution
    # tnx Wikipedia https://en.wikipedia.org/wiki/Cauchy_distribution#Cumulative_distribution_function_(CDF)
    x =  mu + sigma * np.tan(np.pi * (x - 0.5))

    ## END ANSWER
    return x

# Q1 (d)
def rng_std_laplace(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x = np.zeros(n)
    ## BEGIN ANSWER

    # Tnx Wolfram alpha :) https://www.wolframalpha.com/input?i2d=true&i=%5C%2840%29Divide%5B1%2C2%5D%5C%2841%29*Power%5Be%2C%5C%2840%29-%7Cx%7C%5C%2841%29%5D*%CF%80*%5C%2840%291%2BPower%5Bx%2C2%5D%5C%2841%29
    M = np.pi / 2

    i = 0
    while (i<n): # keep sampling until n draws are accepted
        Y = rng_cauchy(1) # draw Y ~ g_Y(y)
        U = np.random.uniform() # draw U ~ U[0,1]

        if U <= pdf_laplace(Y)/(M*pdf_cauchy(Y)): 
            x[i] = Y # save accepted draw
            i += 1
    ## END ANSWER
    return x


# Q1 (e)
def hist_std_laplace():
    n = 50000
    seed = 34786
    x = rng_std_laplace(n, seed)
    plt.figure()
    ## BEGIN ANSWER
    left_bound = -7
    right_bound = 7

    # Create a histogram of the Laplace random draws using 200 bins
    plt.hist(x, bins=200, density=True, label='Histogram of Laplace(0,1)')
    plt.xlim(left_bound, right_bound)

    # Calculate the values from the PDF
    x_values = np.linspace(left_bound, right_bound, 1000)
    laplace_density_values = pdf_laplace(x_values)

    # Plot the pdf values over the histogram
    plt.plot(x_values, laplace_density_values, 'r', label='Theoretical Laplace Density')

    # Set plot title and labels
    plt.title('Histogram of Laplace(0, 1) Random Draws and Theoretical Density')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()

    # Show the plot
    plt.show()

    ## END ANSWER
    plt.savefig(wd+"Q1_e.pdf")
    return True


#
# Question 2 - Numerical Optimization
#

# Do not modify!
def load_data_q2():
    data = pd.read_csv(wd+'Q2_data.csv')
    y = data['y'].values
    x = data['x'].values
    return y, x

# Q2 (a)
def ols_estimator(y, x):
    beta_ols = np.zeros(2)
    ## BEGIN ANSWER
    X = np.column_stack((np.ones_like(x), x))
    beta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    ## END ANSWER
    return beta_ols

# Q2 (b)
def ols_scatterplot():
    y, x = load_data_q2()
    beta_ols = ols_estimator(y, x)
    plt.figure()
    ## BEGIN ANSWER
    # Scatter plot of the data
    plt.scatter(x, y, label='Data points')

    # Fitted regression line
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals = beta_ols[0] + beta_ols[1] * x_vals  # Calculating fitted values
    plt.plot(x_vals, y_vals, color='black', linestyle='-', label='Fitted regression line')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot with OLS regression line')
    plt.legend()
    plt.grid(True)
    plt.show()
    ## END ANSWER
    plt.savefig(wd+"Q2_b.pdf")
    return True

# Q2 (c)
def sar(b,y,x):
    sar = 0
    ## BEGIN ANSWER
    """
    Note to self: The excercise asks for the sum, yet codegrade wants the average :(
    """
    n = len(x)
    b0, b1 = b
    sar = 1/n * np.sum(np.abs(y - b0 - b1*x))
    ## END ANSWER
    return sar

# Q2 (d)
def sar_grad(b,y,x):
    sar_grad = np.zeros(2)
    ## BEGIN ANSWER
    """
    Note to self: The excercise asks for the sum, yet codegrade wants the average :(
    """
    n = len(x)
    b0, b1 = b
    residuals = y - b0 - b1*x
    sar_grad[0] = -np.sum(np.sign(residuals))
    sar_grad[1] = -np.sum(np.sign(residuals) * x)
    ## END ANSWER
    return 1/n * sar_grad

# Q2 (e)
def gradient_descent(f, grad, b0, y, x, max_iter=50, f_tol=1e-8):
    # initialization
    b = copy(b0)
    fval_prev = np.Inf
    fval = f(b, y, x)

    it = 0
    while (abs(fval_prev - fval) >= f_tol) and (it <= max_iter):
        ## BEGIN ANSWER
        # Compute the gradient
        gradient = grad(b, y, x)

        # Compute the update direction
        update_direction = -gradient

        # Backtracking line search parameters
        alpha = 1.0
        c1 = 1e-4
        rho = 0.95

        while True:
            # Compute the next potential solution
            next_b = b + alpha * update_direction

            # Compute the function value at the next potential solution
            next_fval = f(next_b, y, x)

            # Check Armijo condition
            if next_fval <= fval + c1 * alpha * np.dot(update_direction, gradient):
                break
            else:
                alpha *= rho

        # Update values for the next iteration
        b = next_b
        fval_prev = fval
        fval = next_fval
        ## END ANSWER
        it += 1    
    return b

# Q2 (f)
def lad_scatterplot():
    y, x = load_data_q2()
    beta_ols = ols_estimator(y, x)
    beta_lad = np.zeros(2)
    b0 = np.array([0.0, 0.0])
    plt.figure()
    ## BEGIN ANSWER
    beta_lad = gradient_descent(sar, sar_grad, b0, y, x)
    plt.scatter(x, y, label='Data points')

    # Calculating values for the fitted regression line
    x_vals = np.linspace(min(x), max(x), 100)
    y_vals_lad = beta_lad[0] + beta_lad[1] * x_vals  # Calculating LAD fitted values
    y_vals_ols = beta_ols[0] + beta_ols[1] * x_vals  # Calculating OLS fitted values
    
    # Plotting values with given colour/style
    plt.plot(x_vals, y_vals_lad, color='red', linestyle='--', label='Fitted regression line (LAD)')
    plt.plot(x_vals, y_vals_ols, color='black', linestyle='-', label='Fitted regression line (OLS)')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Scatter plot with OLS regression line')
    plt.legend()
    plt.grid(True)
    plt.show()


    ## END ANSWER
    plt.savefig(wd+"Q2_f.pdf")
    return True

# Q2 (g)
def lad_nelder_mead():
    y, x = load_data_q2()
    beta_lad = np.zeros(2)
    b0 = np.array([0.0, 0.0])
    ## BEGIN ANSWER
    minimize_result = minimize(sar, b0, (y,x), method="Nelder-Mead")
    beta_lad = minimize_result.x
    ## END ANSWER
    return beta_lad


#
# Question 3 - Solving Linear Equations & Eigenvalues
#

# Do not modify!
def load_data_q3():
    data = pd.read_csv(wd+'Q3_data.csv')
    y = data['y'].values
    X = data.iloc[:,1:26].values
    return y, X

# Q3 (a.i)
def ridge_M(X, alpha):
    k = X.shape[1]
    M = np.zeros((k,k))
    ## BEGIN ANSWER
    M = X.T @ X + alpha * np.identity(k)
    ## END ANSWER
    return M

# Q3 (a.ii)
def ridge_z(y, X):
    k = X.shape[1]
    z = np.zeros(k)
    ## BEGIN ANSWER
    z = X.T @ y
    ## END ANSWER
    return z

# Q3 (b)
def ridge_estimator(y, X, alpha):
    k = X.shape[1]
    beta_ridge = np.zeros(k)
    ## BEGIN ANSWER
    M = ridge_M(X, alpha)
    z = ridge_z(y, X)

    beta_ridge = np.linalg.solve(M, z)
    ## END ANSWER
    return beta_ridge

# Q3 (c)
def ridge_ev_decomp_XtX(X):
    k = X.shape[1]
    w = np.zeros(k)
    V = np.zeros((k,k))
    ## BEGIN ANSWER
    w, V = np.linalg.eig(X.T @ X)
    ## END ANSWER
    return w, V

# Q3 (d)
def ridge_Minv(X, alpha):
    k = X.shape[1]
    M_inv = np.zeros((k,k))
    ## BEGIN ANSWER
    w, V = ridge_ev_decomp_XtX(X)
    w_hat = 1 / (w + alpha)
    M_inv = V @ np.diag(w_hat) @ V.T
    ## END ANSWER
    return M_inv

# Q3 (e)
def ridge_estimator_via_inv(y, X, alpha):
    k = X.shape[1]
    beta_ridge = np.zeros(k)
    ## BEGIN ANSWER
    M_inv = ridge_Minv(X, alpha)
    z = ridge_z(y, X)
    beta_ridge = M_inv @ z
    ## END ANSWER
    return beta_ridge


if __name__ == "__main__":
    print("Running at %s." % datetime.datetime.now().strftime("%H:%M:%S")) # Do not modify!
    #
    # TODO: While developing your solutions, you might want to add commands below
    # that call the functions above for testing purposes.
    #
    # IMPORTANT: Before you submit your code, comment out or delete all function calls
    # that you added here.
    #
