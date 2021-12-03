
def jacobi_method(A, f, u, N, TOL):
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