import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    # def backprop(act_product: np.ndarray)->np.ndarray:
    #     return _sigmoid(act_product)*(1-_sigmoid(act_product))
    
    # def log_loss(n: int, y: list, probs: list)->float:
    #     return (-1/n)*sum([(yi*np.log(pi)) + ((1-yi)*np.log(1-pi)) 
    #                        for yi, pi in zip[y, probs]
                          # ])
    n_samples, n_features = len(X), len(X[0])
    weights, bias = np.zeros(n_features), 0

    step = 0
    while step<steps:
        step+=1
        z = X@weights + bias
        if step==1:
            print(z)
        probs = _sigmoid(z)
        error_vector = np.subtract(probs, y)
        dldw, dldb = (1/n_samples)*np.dot(X.T, error_vector), (1/n_samples)*np.sum(error_vector)
        update_w, update_b = lr*dldw, lr*dldb
        weights = np.subtract(weights, update_w)
        bias-=update_b

    return weights, bias
    
    