
import numpy as np


def cholesky_decomposition(A):
    """Compute the upper triangular Cholesky factor R such that A = R.T @ R."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    R = np.zeros_like(A)

    for i in range(n):
        for j in range(i, n):
            if i == j:
                sum_k = sum(R[k, i] ** 2 for k in range(i))
                value = A[i, i] - sum_k

                if value <= 0:
                    raise ValueError(
                        "The matrix must be symmetric positive definite."
                    )

                R[i, i] = np.sqrt(value)
            else:
                sum_k = sum(R[k, i] * R[k, j] for k in range(i))
                R[i, j] = (A[i, j] - sum_k) / R[i, i]

    return R


def forward_substitution(L, b):
    """Solve L @ y = b for a lower triangular matrix L."""
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = (
            b[i] - np.dot(L[i, :i], y[:i])
        ) / L[i, i]

    return y


def backward_substitution(R, y):
    """Solve R @ x = y for an upper triangular matrix R."""
    n = len(y)
    x = np.zeros(n)

    for i in reversed(range(n)):
        x[i] = (
            y[i] - np.dot(R[i, i + 1:], x[i + 1:])
        ) / R[i, i]

    return x


def cholesky_solve(A, b):
    """Solve A @ x = b using Cholesky decomposition."""
    R = cholesky_decomposition(A)
    y = forward_substitution(R.T, b)
    x = backward_substitution(R, y)

    return x, R


def is_positive_definite(A):
    """Check whether A is symmetric positive definite."""
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    if not np.allclose(A, A.T):
        return False

    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def main():
    A = np.array([
        [4,  2,  -2,   6],
        [2,  5,   5,   1],
        [-2, 5,  26, -10],
        [6,  1, -10,  12]
    ], dtype=float)

    b = np.array([1, 0, 0, 0], dtype=float)

    if not is_positive_definite(A):
        print("The matrix is not symmetric positive definite.")
        return

    print("The matrix is positive definite. Decomposition is possible.")

    x, R = cholesky_solve(A, b)

    print("\nUpper triangular matrix R (custom):")
    print(np.round(R, 4))

    print("\nSolution Ax = b (custom):")
    print(np.round(x, 4))

    R_numpy = np.linalg.cholesky(A).T

    print("\nUpper triangular matrix R (NumPy):")
    print(np.round(R_numpy, 4))

    y_numpy = np.linalg.solve(R_numpy.T, b)
    x_numpy = np.linalg.solve(R_numpy, y_numpy)

    print("\nSolution Ax = b (NumPy):")
    print(np.round(x_numpy, 4))

    print("\nComparison of solutions:")
    print("Custom solution:", np.round(x, 4))
    print("NumPy solution:", np.round(x_numpy, 4))

    if np.allclose(x, x_numpy):
        print("Solutions are very close.")
    else:
        print("Solutions differ.")

    print("\nResidual ||Ax - b||:")
    print(np.linalg.norm(A @ x - b))


if __name__ == "__main__":
    main()
