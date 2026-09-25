import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    A = np.asarray(A, dtype=float)
    n = len(A)
    augmented = np.hstack((A.copy(), np.eye(n)))

    for col in range(n):

        # Find best pivot row
        pivot_row = col + np.argmax(np.abs(augmented[col:, col]))

        # Singular matrix
        if np.isclose(augmented[pivot_row, col], 0):
            return None

        # Swap rows if necessary
        if pivot_row != col:
            augmented[[col, pivot_row]] = augmented[[pivot_row, col]]

        # Make pivot = 1
        augmented[col] /= augmented[col, col]

        # Make every other entry in this column = 0
        for row in range(n):
            if row != col:
                augmented[row] -= (
                    augmented[row, col] * augmented[col]
                )

    # Right half is now A⁻¹
    return augmented[:, n:]