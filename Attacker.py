import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras import models
import pickle

def create_and_train_model(dataset_path="Attacker_dataset.pkl", model_path = "Attacker_trained_model.keras", epochs=500, batch_size=128):
    """
    Crea e addestra un modello su un dataset.
    Args:
        dataset_path (str): Percorso al dataset in formato pickle.
        model_path (str): Percorso per salvare il modello addestrato.
        epochs (int): Numero di epoche di addestramento.
        batch_size (int): Dimensione del batch.
    """
    # Carica il dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Prepara i dati
    inputs = []
    outputs = []

    for row in dataset:
        inputs.append(row["inputs"])
        outputs.append(row["best_output"])

    inputs = np.array(inputs)  # [num_samples, 20, 2]
    outputs = np.array(outputs)  # [num_samples, 10, 2]

    # Definisci il modello
    model = Sequential([
        Input(shape=(20, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(20, activation='relu'),  # 10 coppie x 2 dimensioni (throttle, brake)
        tf.keras.layers.Reshape((10, 2))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Addestra il modello
    print("Inizio addestramento del modello...")
    model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    print("Addestramento completato!")

    # Salva il modello
    model.save(model_path, save_format="keras")
    print(f"Modello salvato in: {model_path}")

def load_model(model_path="Attacker_trained_model.keras"):
    """
    Carica il modello addestrato.
    Args:
        model_path (str): Percorso al file del modello salvato.
    Returns:
        tensorflow.keras.Model: Modello caricato.
    """
    return tf.keras.models.load_model(model_path)

def attack(input_sequence, model_path = "Attacker_trained_model.keras"):
    """
    Esegue un attacco predittivo dato un input di 20 coppie.
    Args:
        input_sequence (numpy.ndarray): Sequenza di input [20, 2].
        model_path (str): Percorso al file del modello addestrato.
    Returns:
        numpy.ndarray: Predizione di output [10, 2].
    """
    # Verifica la dimensione dell'input
    if input_sequence.shape != (20, 2):
        raise ValueError("La sequenza di input deve avere dimensione (20, 2)")

    # Carica il modello
    model = load_model(model_path)

    # Esegui la predizione
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Aggiunge dimensione batch
    predicted_output = model.predict(input_sequence)

    # Rimuovi la dimensione batch
    return predicted_output[0]

if __name__ == "__main__":
    # Per il training del modello
    dataset_path = "Attacker_dataset.pkl"
    model_path = "Attacker_trained_model.keras"

    # Crea e addestra il modello
    create_and_train_model(dataset_path=dataset_path, model_path=model_path, epochs=500, batch_size=128)

    # Test della funzione attack
    throttle_mean = 50
    brake_mean = 200

    # Genera una sequenza di input di esempio
    input_sequence = np.random.multivariate_normal(
        mean=[throttle_mean, brake_mean],
        cov=[[100, 0], [0, 5000]],
        size=20
    )
    input_sequence[:, 0] = np.clip(input_sequence[:, 0], 0, 100)
    input_sequence[:, 1] = np.clip(input_sequence[:, 1], 0, 1000)

    # Predizione usando il modello addestrato
    output_sequence = attack(input_sequence, model_path=model_path)

    # Stampa input e output
    print("Input Sequence (20 coppie):")
    print(input_sequence)
    print("\nPredicted Output Sequence (10 coppie):")
    print(output_sequence)

'''import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
import pickle
import psutil
import time
import matplotlib.pyplot as plt

def create_and_train_model(dataset_path="Attacker_dataset.pkl", model_path="Attacker_trained_model.keras", epochs=500, batch_size=128):
    """
    Crea e addestra un modello su un dataset monitorando CPU, RAM e tempo.
    Args:
        dataset_path (str): Percorso al dataset in formato pickle.
        model_path (str): Percorso per salvare il modello addestrato.
        epochs (int): Numero di epoche di addestramento.
        batch_size (int): Dimensione del batch.
    """
    # Monitoraggio iniziale
    process = psutil.Process()
    start_time = time.time()

    # Liste per monitoraggio
    cpu_usage = []
    ram_usage = []
    timestamps = []

    def log_usage():
        """Registra l'uso di CPU e RAM."""
        cpu_usage.append(psutil.cpu_percent(interval=None))
        ram_usage.append(process.memory_info().rss / (1024 * 1024))  # Convert RAM in MB
        timestamps.append(time.time() - start_time)

    # Carica il dataset
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Prepara i dati
    inputs = []
    outputs = []

    for row in dataset:
        inputs.append(row["inputs"])
        outputs.append(row["best_output"])

    inputs = np.array(inputs)  # [num_samples, 20, 2]
    outputs = np.array(outputs)  # [num_samples, 10, 2]

    # Definisci il modello
    model = Sequential([
        Input(shape=(20, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(20, activation='relu'),  # 10 coppie x 2 dimensioni (throttle, brake)
        tf.keras.layers.Reshape((10, 2))
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Callback per monitorare CPU e RAM a ogni epoca
    class MonitorCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log_usage()

    # Addestra il modello
    print("Inizio addestramento del modello...")
    model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[MonitorCallback()])
    print("Addestramento completato!")

    # Salva il modello
    model.save(model_path, save_format="keras")
    print(f"Modello salvato in: {model_path}")

    # Monitoraggio finale
    elapsed_time = time.time() - start_time

    # Plotta i dati di monitoraggio
    plt.figure(figsize=(12, 6))

    # CPU Usage
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, cpu_usage, label="CPU Usage (%)", color="blue")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage During Training")
    plt.legend()

    # RAM Usage
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, ram_usage, label="RAM Usage (MB)", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("RAM Usage (MB)")
    plt.title("RAM Usage During Training")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Stampa il tempo totale
    print(f"\nTempo totale per l'addestramento: {elapsed_time:.2f} secondi")

def load_model(model_path="Attacker_trained_model.keras"):
    """
    Carica il modello addestrato.
    Args:
        model_path (str): Percorso al file del modello salvato.
    Returns:
        tensorflow.keras.Model: Modello caricato.
    """
    return tf.keras.models.load_model(model_path)

def attack(input_sequence, model_path="Attacker_trained_model.keras"):
    """
    Esegue un attacco predittivo dato un input di 20 coppie.
    Args:
        input_sequence (numpy.ndarray): Sequenza di input [20, 2].
        model_path (str): Percorso al file del modello addestrato.
    Returns:
        numpy.ndarray: Predizione di output [10, 2].
    """
    # Verifica la dimensione dell'input
    if input_sequence.shape != (20, 2):
        raise ValueError("La sequenza di input deve avere dimensione (20, 2)")

    # Carica il modello
    model = load_model(model_path)

    # Esegui la predizione
    input_sequence = np.expand_dims(input_sequence, axis=0)  # Aggiunge dimensione batch
    predicted_output = model.predict(input_sequence)

    # Rimuovi la dimensione batch
    return predicted_output[0]

if __name__ == "__main__":
    # Per il training del modello
    dataset_path = "Attacker_dataset.pkl"
    model_path = "Attacker_trained_model.keras"

    # Crea e addestra il modello
    create_and_train_model(dataset_path=dataset_path, model_path=model_path, epochs=500, batch_size=128)'''