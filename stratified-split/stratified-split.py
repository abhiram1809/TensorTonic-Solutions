import numpy as np

def stratified_split(X: list, y: list, test_size: float = 0.2, seed: int = 42) -> dict:
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)

    train_indices, test_indices = [], []

    for cls in np.unique(y):
        indices = np.flatnonzero(y == cls)
        rng.shuffle(indices)

        split = round(len(indices) * test_size)

        test_indices.extend(indices[:split])
        train_indices.extend(indices[split:])

    train_indices = np.sort(train_indices)
    test_indices = np.sort(test_indices)

    return {
        "X_train": X[train_indices],
        "X_test": X[test_indices],
        "y_train": y[train_indices],
        "y_test": y[test_indices]
    }