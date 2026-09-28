def gradient_descent_quadratic(a: float, b: float, c: float, x0: float, lr: float, steps: int) -> float:
    """
    Returns the final scalar x after the requested iterations.
    """
    # derivative = 2*a*x0 + b
    xi=x0
    i = 0
    while i<steps: 
        # if derivative < 0:
        #     lr = -(lr)
        #     x0 += lr
        # else:
        #     lr = lr
        #     x0 += lr
        
        # derivative = 2*a*x0 + b
        xi = xi - lr * (2*a*xi + b)
        i+=1

    return xi
    