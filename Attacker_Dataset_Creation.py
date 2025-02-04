import numpy as np
import matlab.engine
import os
import pickle
import time
import psutil  # Per monitorare l'uso di CPU e RAM

global eng

def generate_random_couples(target_mean, covariance_matrix, size=100):
    """
    Genera coppie random che rispettano i vincoli di media e varianza.
    Args:
        target_mean (list): Lista contenente [media_throttle, media_brake].
        covariance_matrix (list): Matrice di covarianza.
        size (int): Numero di coppie da generare.
    Returns:
        numpy.ndarray: Coppie generate [size, 2].
    """
    random_couples = np.random.multivariate_normal(mean=target_mean, cov=covariance_matrix, size=size)
    random_couples[:, 0] = np.clip(random_couples[:, 0], 0, 100)  # throttle (0-100)
    random_couples[:, 1] = np.clip(random_couples[:, 1], 0, 1000)  # brake (0-1000)
    return random_couples

def simulate_speed(sequence):
    """
    Simula il guadagno di velocità per una sequenza di input-output.
    Args:
        sequence (numpy.ndarray): Sequenza di 30 coppie [30, 2].
        eng (matlab.engine.MatlabEngine): Instanza del MATLAB Engine.
    Returns:
        float: Delta speed.
    """
    try:
        throttle_input = matlab.double(sequence[:, 0].tolist())
        brake_input = matlab.double(sequence[:, 1].tolist())
        delta_speed = eng.Attacker_simulate_speed(throttle_input, brake_input, nargout=1)
    except Exception as e:
        print(f"Errore durante la simulazione: {e}")
        delta_speed = -1e6  # Penalità in caso di errore
    return delta_speed

def generate_dataset(output_file="Attacker_dataset.pkl", num_samples=10000, random_candidates=50, save_every=50):
    """
    Genera un dataset per il modello basato su input-output ottimizzati.
    Salva progressivamente ogni 50 righe.
    
    Args:
        output_file (str): Percorso del file per salvare il dataset.
        num_samples (int): Numero totale di righe del dataset.
        random_candidates (int): Numero di candidati casuali generati per ogni input.
        save_every (int): Frequenza con cui salvare il dataset.
    """
    dataset = []
    existing_rows = 0

    # Carica il dataset esistente se presente
    if os.path.exists(output_file):
        with open(output_file, "rb") as f:
            dataset = pickle.load(f)
            existing_rows = len(dataset)
            print(f"Caricato dataset esistente con {existing_rows} righe.")

    # Calcola il numero di righe rimanenti da generare
    rows_to_generate = num_samples - existing_rows
    print(f"Generazione di {rows_to_generate} righe.")

    for row_idx in range(rows_to_generate):
        # Imposta il seed basato sul tempo reale
        seed = int(time.time())
        np.random.seed(seed)
        print(f"Seed impostato: {seed}")

        # Genera input casuale
        throttle_mean = np.random.uniform(30, 80)
        brake_mean = np.random.uniform(100, 350)

        inputs = np.random.multivariate_normal(
            mean=[throttle_mean, brake_mean],
            cov=[[100, 0], [0, 5000]],
            size=20
        )
        inputs[:, 0] = np.clip(inputs[:, 0], 0, 100)
        inputs[:, 1] = np.clip(inputs[:, 1], 0, 1000)

        # Genera candidati casuali per l'output
        candidate_outputs = np.random.multivariate_normal(
            mean=[throttle_mean, brake_mean],
            cov=[[100, 0], [0, 5000]],
            size=random_candidates * 10  # Ogni candidato dovrebbe contenere 10 coppie
        )
        candidate_outputs[:, 0] = np.clip(candidate_outputs[:, 0], 0, 100)
        candidate_outputs[:, 1] = np.clip(candidate_outputs[:, 1], 0, 1000)

        # Trova l'output che massimizza il delta_speed
        max_speed = float('-inf')
        best_output = None

        for i in range(random_candidates):
            start_idx = i * 10
            end_idx = (i + 1) * 10
            candidate = candidate_outputs[start_idx:end_idx]
            
            if candidate.shape != (10, 2):
                print(f"Errore: dimensione errata per candidato: {candidate.shape}")
                continue

            # Combina input e candidato in una sequenza
            sequence = np.vstack((inputs, candidate))
            delta_speed = simulate_speed(sequence)

            if delta_speed > max_speed:
                max_speed = delta_speed
                best_output = candidate

        if best_output is None:
            print(f"Nessun candidato valido trovato per la riga {row_idx + 1}")
            continue

        # Aggiungi la riga al dataset
        dataset.append({
            "inputs": inputs.tolist(),
            "best_output": best_output.tolist(),
            "max_speed": max_speed,
            "input_throttle_mean": throttle_mean,
            "input_brake_mean": brake_mean
        })

        print(f"Riga aggiunta al dataset: {row_idx + 1} (delta_speed: {max_speed:.2f})")

        # Salvataggio progressivo ogni `save_every` righe
        if (row_idx + 1) % save_every == 0 or (row_idx + 1) == rows_to_generate:
            with open(output_file, "wb") as f:
                pickle.dump(dataset, f)
            print(f"Salvate {row_idx + 1 + existing_rows} righe su {num_samples} totali.")

    print("Dataset completato e salvato.")

# Esegui la generazione del dataset
if __name__ == "__main__":
    eng = matlab.engine.start_matlab()
    print("Inizio generazione dataset...")
    dataset = generate_dataset(num_samples=10000, random_candidates=50, output_file="Attacker_dataset.pkl")
    print("Dataset generato e salvato con successo.")
    eng.quit()

'''import numpy as np
import matlab.engine
import psutil
import time
import matplotlib.pyplot as plt

global eng

def generate_single_row():
    """
    Genera una singola riga per il dataset, monitorando continuamente CPU, RAM e tempo.
    Returns:
        dict: Contiene i dati di CPU, RAM e tempo utilizzati per la generazione.
    """
    # Monitoraggio iniziale
    process = psutil.Process()
    start_time = time.time()

    # Liste per registrare CPU, RAM e timestamp
    cpu_usage = []
    ram_usage = []
    timestamps = []

    # Funzione per registrare l'uso corrente
    def log_usage():
        cpu_usage.append(psutil.cpu_percent(interval=None))
        ram_usage.append(process.memory_info().rss / (1024 * 1024))  # Convertito in MB
        timestamps.append(time.time() - start_time)

    # Imposta il seed basato sul tempo reale
    seed = int(time.time())
    np.random.seed(seed)

    # Genera input casuale
    throttle_mean = np.random.uniform(30, 80)
    brake_mean = np.random.uniform(100, 350)

    inputs = np.random.multivariate_normal(
        mean=[throttle_mean, brake_mean],
        cov=[[100, 0], [0, 5000]],
        size=20
    )
    inputs[:, 0] = np.clip(inputs[:, 0], 0, 100)
    inputs[:, 1] = np.clip(inputs[:, 1], 0, 1000)

    log_usage()  # Log dopo la generazione degli input

    # Genera candidati casuali per l'output
    random_candidates = 50
    candidate_outputs = np.random.multivariate_normal(
        mean=[throttle_mean, brake_mean],
        cov=[[100, 0], [0, 5000]],
        size=random_candidates * 10  # Ogni candidato dovrebbe contenere 10 coppie
    )
    candidate_outputs[:, 0] = np.clip(candidate_outputs[:, 0], 0, 100)
    candidate_outputs[:, 1] = np.clip(candidate_outputs[:, 1], 0, 1000)

    log_usage()  # Log dopo la generazione dei candidati

    # Trova l'output che massimizza il delta_speed
    max_speed = float('-inf')
    best_output = None

    for i in range(random_candidates):
        start_idx = i * 10
        end_idx = (i + 1) * 10
        candidate = candidate_outputs[start_idx:end_idx]

        if candidate.shape != (10, 2):
            continue

        # Combina input e candidato in una sequenza
        sequence = np.vstack((inputs, candidate))
        delta_speed = simulate_speed(sequence)

        if delta_speed > max_speed:
            max_speed = delta_speed
            best_output = candidate

        log_usage()  # Log durante l'elaborazione dei candidati

    if best_output is None:
        print("Nessun candidato valido trovato.")
        return None

    # Monitoraggio finale
    elapsed_time = time.time() - start_time

    # Risultati
    return {
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage,
        "timestamps": timestamps,
        "elapsed_time": elapsed_time,
        "inputs": inputs.tolist(),
        "best_output": best_output.tolist(),
        "max_speed": max_speed
    }

def simulate_speed(sequence):
    """
    Simula il guadagno di velocità per una sequenza di input-output.
    Args:
        sequence (numpy.ndarray): Sequenza di 30 coppie [30, 2].
    Returns:
        float: Delta speed.
    """
    try:
        throttle_input = matlab.double(sequence[:, 0].tolist())
        brake_input = matlab.double(sequence[:, 1].tolist())
        delta_speed = eng.Attacker_simulate_speed(throttle_input, brake_input, nargout=1)
    except Exception as e:
        print(f"Errore durante la simulazione: {e}")
        delta_speed = -1e6  # Penalità in caso di errore
    return delta_speed

if __name__ == "__main__":
    # Avvia MATLAB Engine
    eng = matlab.engine.start_matlab()
    print("Inizio monitoraggio per una singola riga...")

    # Genera una singola riga e monitora
    results = generate_single_row()

    # Termina MATLAB Engine
    eng.quit()

    if results:
        print("\nRisultati del monitoraggio:")
        print(f"Elapsed Time: {results['elapsed_time']:.2f} seconds")
        print(f"Max Speed: {results['max_speed']}")

        # Plot CPU e RAM
        plt.figure(figsize=(12, 6))

        # Plot CPU Usage
        plt.subplot(2, 1, 1)
        plt.plot(results["timestamps"], results["cpu_usage"], label="CPU Usage (%)", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage Over Time")
        plt.legend()

        # Plot RAM Usage
        plt.subplot(2, 1, 2)
        plt.plot(results["timestamps"], results["ram_usage"], label="RAM Usage (MB)", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("RAM Usage (MB)")
        plt.title("RAM Usage Over Time")
        plt.legend()

        plt.tight_layout()
        plt.show()'''