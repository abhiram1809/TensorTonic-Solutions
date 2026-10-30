import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    dictionary = {word: 0 for index, word in enumerate(vocab)}
    
    for token in tokens:
        if token in dictionary:
            dictionary[token] +=1
    
    return np.asarray(list(dictionary.values()))