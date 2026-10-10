import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X, y = np.asarray(X), np.asarray(y)
    rando = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rando.shuffle(indices)

    X, y = X[indices], y[indices]
    i = 0
    print(X)
    while i<len(y): 
        if drop_last and i + batch_size > len(y):
            # print("Reached, Drop Last")
            break
        # print(X[i:i+batch_size])
        yield np.asarray(X[i:i+batch_size]),np.asarray(y[i:i+batch_size])
        i+=batch_size
        print(i)