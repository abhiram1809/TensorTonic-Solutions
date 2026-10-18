import math

def log_loss(y_true: list, y_pred: list, eps: float = 1e-15) -> list:
    """
    Returns a list of loss values.
    """
    def compute_pcap(p):
        return min(1-eps, max(eps, p))
    losses = []
    for y_t, y_p in zip(y_true, y_pred):
        pcap = compute_pcap(y_p)
        # print(pcap, y_t, y_p)
        loss = -1*(y_t*math.log(pcap) + (1-y_t)*math.log(1-pcap))
        losses.append(loss)
    return losses