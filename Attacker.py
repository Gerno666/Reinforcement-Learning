import pickle
import numpy as np

def attack(model_path="Attacker_optimized_model.pkl"):
    """
    Genera 10 coppie di throttle e brake utilizzando un modello addestrato.
    Returns:
        numpy.ndarray: Predizioni [10, 2] (throttle, brake).
    """

    # Carica il modello ottimizzato
    with open(model_path, "rb") as f:
        optimized_pairs = pickle.load(f)

    # Restituisci le coppie ottimizzate
    return optimized_pairs