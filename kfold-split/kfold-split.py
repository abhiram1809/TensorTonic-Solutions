import numpy as np

def kfold_split(N: int, k: int, shuffle: bool = True, seed: int = 0) -> list:
    """
    Returns a list of dictionaries with train_idx and val_idx.
    """
    def build_dict(batches: int, fold: int)->dict:
        build = {"train_idx":[], "val_idx": batches[fold-1]}
        for i in range(1, len(batches)+1):
            if i!=fold:
                build["train_idx"] += list(batches[i-1])
        build["train_idx"] = np.asarray(build["train_idx"])
        build["val_idx"] = np.asarray(build["val_idx"])
        return build
            
    rando = np.random.default_rng(seed)
    indices = np.arange(N)
    if shuffle:
        rando.shuffle(indices)
    max_batch = round(N/k) +1 if N%k!=0 else N/k
    normal_batch = round(N/k) if N%k!=0 else N/k
    print(max_batch, normal_batch)
    batches = []
    i=0
    while i<N:
        if i==0:
            batches.append(indices[i:int(max_batch)])
            i+=int(max_batch)
            continue
        else:    
            batches.append(indices[i:i+int(normal_batch)])
            i+=int(normal_batch)
    k_folds = []
    for j in range(1, k+1):
        k_folds.append(build_dict(batches, j))
    return k_folds
    
    
    