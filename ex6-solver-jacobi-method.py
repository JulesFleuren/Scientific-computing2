from matplotlib import pyplot as plt

def jacobi_method(A, f, u, TOL):
    """
    Solves the linear system Ax = f using the Jacobi method.
    """
    # Initialize the solution vector.
    x = [0] * len(u)
    # Initialize the error vector.
    e = [0] * len(u)
    # Initialize the error.
    error = 1
    # Initialize the iteration counter.
    k = 0
    # Iterate until the error is less than the tolerance.
    while error > TOL:
        # Update the iteration counter.
        k += 1
        # Update the solution vector.
        for i in range(len(u)):
            x[i] = (f[i] - sum(A[i][j] * x[j] for j in range(len(u)))) / A[i][i]
        # Update the error vector.
        for i in range(len(u)):
            e[i] = abs(x[i] - u[i])
        # Update the error.
        error = max(e)
    # Return the solution vector and the number of iterations.
    return x, k

def plot_scaled_residual(A, f, u, TOL):
    """
    Plots the scaled residual for the Jacobi method.
    """
    # Solve the linear system.
    x, k = jacobi_method(A, f, u, TOL)
    # Initialize the scaled residual vector.
    r = [0] * len(u)
    # Compute the scaled residual.
    for i in range(len(u)):
        r[i] = abs(u[i] - x[i]) / abs(u[i])
    # Plot the scaled residual.
    plt.plot(range(k), r)
    plt.xlabel('Iteration')
    plt.ylabel('Scaled Residual')
    plt.show()