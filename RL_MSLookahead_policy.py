import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Carica il modello addestrato
model_path = 'cost_predictor_model_R.h5'
dataset_path = 'RL_10k_simulations_data_normalized_update.csv'

# Variabile globale per mantenere il modello in memoria
model = None

# Variabile globale per accumulare le coppie
training_buffer = []

def load_or_initialize_model():
    global model
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Il file del modello {model_path} non esiste.")
        model = load_model(model_path)
        print("Modello caricato in memoria.")
    return model

# Funzione per normalizzare lo stato
def normalize_state(state_vector):
    """
    Normalizza lo stato basandosi sui range utilizzati in fase di addestramento.
    """
    # Converte lo stato in array NumPy se necessario
    if not isinstance(state_vector, np.ndarray):
        state_vector = np.array(state_vector, dtype=np.float32)

    # Applica la normalizzazione
    state_vector[0] /= 120.0  # Speed_1_min_ago
    state_vector[1] /= 4500.0  # RPM_1_min_ago
    state_vector[2:62] /= 100.0 # Throttle
    state_vector[62:122] /= 1000.0 # Brake
    state_vector[122] /= 120.0  # Speed_now
    state_vector[123] /= 4500.0  # RPM_now

    return state_vector

# Funzione per calcolare il costo di uno stato
def evaluate_cost(state_vector):
    """
    Calcola il costo dello stato utilizzando il modello di rete neurale.
    """
    # Converte lo stato in un array NumPy se necessario
    if not isinstance(state_vector, np.ndarray):
        state_vector = np.array(state_vector, dtype=np.float32)

    # Normalizza lo stato
    state_norm = normalize_state(state_vector)

    # Ridimensiona l'input per il modello: (1, num_features)
    state_norm = state_norm.reshape(1, -1)

    # Carica il modello
    model = load_or_initialize_model()

    # Predici il cost_to_go normalizzato
    cost_to_go_norm = model.predict(state_norm, verbose=0)  # questo è normalizzato tra 0 e 1 se hai addestrato così

    # cost_to_go_norm è un array Nx1, noi vogliamo un singolo valore
    cost_to_go_norm_value = float(cost_to_go_norm[0,0])

    # Denormalizzazione del cost_to_go
    # Durante il training il costo era normalizzato dividendo per 100, quindi qui moltiplichiamo per 100
    cost_value = cost_to_go_norm_value * 200.0
    print(cost_value)
    return cost_value


def train_policy(state_vector, cost):
    """
    Riaddestramento periodico del modello utilizzando l'intero dataset aggiornato.
    """
    global training_buffer  # Buffer per accumulare nuove coppie

    # Converte lo stato in un array NumPy se necessario
    if not isinstance(state_vector, np.ndarray):
        state_vector = np.array(state_vector, dtype=np.float32)

    # Normalizza lo stato e il costo
    state_norm = normalize_state(state_vector)
    cost_norm = cost / 200.0

    # Aggiungi la nuova coppia al buffer
    training_buffer.append((state_norm, cost_norm))
    print(f"Coppia aggiunta al buffer. Dimensione attuale del buffer: {len(training_buffer)}")

    # Salva la nuova coppia nel dataset aggiornato
    new_data = np.hstack([state_norm, cost_norm]).reshape(1, -1)
    df = pd.DataFrame(new_data)
    df.to_csv(dataset_path, mode='a', header=False, index=False)
    print("Nuova coppia aggiunta al dataset e salvata.")

    # Riaddestramento periodico ogni 500 nuove coppie
    if len(training_buffer) >= 250:
        print("Buffer pieno. Inizio del riaddestramento periodico con l'intero dataset...")

        # Carica il modello
        model = load_or_initialize_model()
        print("Modello caricato per riaddestramento periodico.")

        # Carica l'intero dataset aggiornato
        dataset = pd.read_csv(dataset_path, header=None)
        X = dataset.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
        y = dataset.iloc[:, -1].values  # Ultima colonna

        # Dividi i dati in training e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ricompilazione del modello
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        # Riaddestramento del modello con l'intero dataset
        print("Riaddestramento del modello con l'intero dataset aggiornato...")
        model.fit(X_train, y_train, epochs=200, batch_size=64, validation_data=(X_test, y_test), verbose=1)

        # Salva il modello aggiornato
        model.save(model_path)
        print("Modello riaddestrato e salvato come cost_predictor_model_R.h5")

        # Svuota il buffer
        training_buffer = []
        print("Buffer svuotato dopo il riaddestramento.")