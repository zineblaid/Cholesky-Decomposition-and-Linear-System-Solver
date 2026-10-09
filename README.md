# Cholesky Decomposition and Linear System Solver

## Overview

This project implements the **Cholesky decomposition method** in Python to solve a system of linear equations of the form \(Ax=b\).

The implementation computes the upper triangular matrix \(R\), then solves the system using forward and backward substitution. The results are compared with NumPy's built-in linear algebra functions.

## Features

- Check whether a matrix is positive definite.
- Implement Cholesky decomposition from scratch.
- Solve triangular systems using forward and backward substitution.
- Solve a complete linear system \(Ax=b\).
- Compare custom results with NumPy's implementation.
- Verify numerical agreement using `numpy.allclose()`.

## Mathematical Background

For a symmetric positive-definite matrix \(A\), Cholesky decomposition expresses the matrix as:

\[
A = R^T R
\]

where \(R\) is an upper triangular matrix.

To solve \(Ax=b\), the system is divided into two triangular systems:

1. Forward substitution: \(R^T y=b\)
2. Backward substitution: \(Rx=y\)

The final solution is the vector \(x\).

## Technologies Used

- Python
- NumPy

## Installation

Install NumPy using:

```bash
pip install numpy
```

## How to Run

Run the Python script:

```bash
python main.py
```

## Example

The program uses the following matrix:

\[
A =
\begin{bmatrix}
4 & 2 & -2 & 6 \\
2 & 5 & 5 & 1 \\
-2 & 5 & 26 & -10 \\
6 & 1 & -10 & 12
\end{bmatrix}
\]

and the vector:

\[
b =
\begin{bmatrix}
1 \\ 0 \\ 0 \\ 0
\end{bmatrix}
\]

The program computes the upper triangular matrix \(R\), solves the system, and compares the custom solution with NumPy's result.

## Project Structure

```text
cholesky-decomposition/
├── main.py
├── README.md
└── .gitignore
```

## Learning Objectives

- Understand Cholesky factorization.
- Implement numerical algorithms using Python and NumPy.
- Apply forward and backward substitution.
- Compare a custom numerical method with a trusted library implementation.

## Conclusion

This project demonstrates how Cholesky decomposition can be used to solve linear systems efficiently when the coefficient matrix is symmetric and positive definite.

The custom implementation is validated against NumPy to check the accuracy of the computed solution.
