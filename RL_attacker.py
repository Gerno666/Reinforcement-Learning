'''import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pickle
import os

# Disabilita i log dettagliati di TensorFlow
tf.get_logger().setLevel('ERROR')

# Path per salvare i pesi ottimizzati
MODEL_WEIGHTS_FILE = "RL_optimized_model_weights.pkl"

# Media e covarianza richieste
THROTTLE_MEAN = 50
BRAKE_MEAN = 500
COVARIANCE_MATRIX = np.array([[50, -600], [-600, 2000]])

def create_model():
    """
    Crea un modello di rete neurale con 3 layer.
    """
    model = Sequential([
        Dense(10, input_shape=(20,), activation='relu'),  # Layer nascosto 1
        Dense(5, activation='relu'),                     # Layer nascosto 2
        Dense(2)  # Layer di output per throttle e brake
    ])
    return model

def optimize_neural_network_weights(single_state):
    """
    Ottimizza i pesi della rete neurale utilizzando Nevergrad.
    """
    import nevergrad as ng
    import matlab.engine
    import gc

    # Avvia il motore MATLAB
    eng = matlab.engine.start_matlab()

    # Crea il modello
    model = create_model()
    expected_weights = model.get_weights()
    total_weights = sum(w.size for w in expected_weights)

    print(f"Numero di pesi da ottimizzare: {total_weights}")

    # Configura l'ottimizzatore Nevergrad
    optimizer = ng.optimizers.CMA(parametrization=total_weights, budget=1000, num_workers=1)

    # Variabile per il conteggio delle iterazioni
    iteration_count = 0

    def objective(weights):
        """
        Funzione obiettivo per ottimizzare i pesi.
        """
        nonlocal iteration_count
        iteration_count += 1  # Incrementa il contatore delle iterazioni

        reshaped_weights = []
        start = 0
        for w in expected_weights:
            size = w.size
            reshaped_weights.append(np.array(weights[start:start + size]).reshape(w.shape))
            start += size
        model.set_weights(reshaped_weights)

        # Converti lo stato in array numpy
        state_np = np.array(single_state).reshape(1, -1)

        # Predici throttle e brake
        prediction = model.predict(state_np)
        predicted_throttle, predicted_brake = prediction[0]

        # Genera valori rispettando media e covarianza
        adjusted_inputs = np.random.multivariate_normal(
            [THROTTLE_MEAN, BRAKE_MEAN], COVARIANCE_MATRIX, size=1
        )[0]

        # Normalizzazione dei valori rispetto ai loro range massimi
        normalized_throttle = adjusted_inputs[0] / 100  # throttle normalizzato su [0, 1]
        normalized_brake = adjusted_inputs[1] / 1000    # brake normalizzato su [0, 1]
        
        # Introduzione di un fattore casuale
        random_factor = np.random.rand()  # Numero casuale tra 0 e 1
        
        # Applicazione del vincolo con casualità
        if random_factor < 0.3:
            # 30% delle volte assegna solo throttle
            throttle = max(0, min(100, round(normalized_throttle * 100)))  # throttle limitato a [0, 100]
            brake = 0
        elif random_factor > 0.7:
            # 30% delle volte assegna solo brake
            throttle = 0
            brake = max(0, min(1000, round(normalized_brake * 1000)))  # brake limitato a [0, 1000]
        else:
            # 40% delle volte utilizza i valori normalizzati per decidere
            if normalized_throttle > normalized_brake:
                # throttle normalizzato è maggiore, usa throttle
                throttle = max(0, min(100, round(normalized_throttle * 100)))
                brake = 0
            else:
                # brake normalizzato è maggiore, usa brake
                throttle = 0
                brake = max(0, min(1000, round(normalized_brake * 1000)))


        # Simula e calcola la violazione
        try:
            average_violation = eng.run_simulation_in_simulink(throttle, brake)
            if average_violation is not None:
                print(f"Iterazione {iteration_count}: Throttle={throttle}, Brake={brake}, Violazione={average_violation}")
                return -average_violation  # Minimizza la violazione
        except Exception as e:
            print(f"Errore durante la simulazione all'iterazione {iteration_count}: {e}")
            return 0  # Nessuna penalità per errori

        return 0

    # Esegui l'ottimizzazione
    try:
        recommendation = optimizer.minimize(objective)
        best_weights = recommendation.value

        # Salva i pesi ottimizzati
        with open(MODEL_WEIGHTS_FILE, "wb") as f:
            pickle.dump(best_weights, f)

        print("Pesi ottimizzati salvati.")
    except Exception as e:
        print(f"Errore durante l'ottimizzazione: {e}")
    finally:
        eng.quit()
        gc.collect()

def calculate_optimal_controls(state):
    """
    Predice i valori ottimali di throttle e brake utilizzando i pesi ottimizzati.
    """
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        print("Pesi ottimizzati non trovati. Avvio dell'ottimizzazione...")
        optimize_neural_network_weights(state)

    # Crea il modello
    model = create_model()

    # Carica i pesi ottimizzati
    with open(MODEL_WEIGHTS_FILE, "rb") as f:
        best_weights = pickle.load(f)

    # Imposta i pesi ottimizzati nel modello
    reshaped_weights = []
    start = 0
    expected_weights = model.get_weights()
    for w in expected_weights:
        size = w.size
        reshaped_weights.append(np.array(best_weights[start:start + size]).reshape(w.shape))
        start += size
    model.set_weights(reshaped_weights)

    # Predici throttle e brake
    state_np = np.array(state).reshape(1, -1)
    prediction = model.predict(state_np)
    predicted_throttle, predicted_brake = prediction[0]

    print(predicted_throttle)

    # Normalizzazione dei valori rispetto ai loro range massimi
    normalized_throttle = predicted_throttle / 100  # throttle normalizzato su [0, 1]
    normalized_brake = predicted_brake / 1000    # brake normalizzato su [0, 1]
    
    # Introduzione di un fattore casuale
    random_factor = np.random.rand()  # Numero casuale tra 0 e 1
    
    # Applicazione del vincolo con casualità
    if random_factor < 0.3:
        # 30% delle volte assegna solo throttle
        throttle = max(0, min(100, round(normalized_throttle * 100)))  # throttle limitato a [0, 100]
        brake = 0
    elif random_factor > 0.7:
        # 30% delle volte assegna solo brake
        throttle = 0
        brake = max(0, min(1000, round(normalized_brake * 1000)))  # brake limitato a [0, 1000]
    else:
        # 40% delle volte utilizza i valori normalizzati per decidere
        if normalized_throttle > normalized_brake:
            # throttle normalizzato è maggiore, usa throttle
            throttle = max(0, min(100, round(normalized_throttle * 100)))
            brake = 0
        else:
            # brake normalizzato è maggiore, usa brake
            throttle = 0
            brake = max(0, min(1000, round(normalized_brake * 1000)))

    return throttle, brake

# Variabili globali per mantenere lo stato dell'attacco
attack_list = []
current_index = 0

def reset_globals():
    """
    Resetta le variabili globali attack_list e current_index.
    """
    global attack_list, current_index
    attack_list = []  # Reset della lista
    current_index = 0  # Reset dell'indice
    print("Variabili Python resettate: attack_list e current_index")

# Matrice di covarianza base
BASE_COVARIANCE_MATRIX = np.array([[200, -600], [-600, 2000]])
THROTTLE_CLIP = (0, 100)
BRAKE_CLIP = (0, 1000)

def calculate_optimal_controls(state):
    """
    Genera una finestra di 500 valori di throttle e brake basata sullo stato iniziale.
    L'attacco diventa progressivamente più aggressivo:
    - Se delta V < 0: Incrementa la media di throttle per i primi 20 valori.
    - Se throttle attivi < brake attivi: Incrementa probabilità di throttle > 0 nei primi 20 valori.
    """
    global attack_list, current_index

    # Inizializza se necessario
    if current_index == 0:

        attack_list = []

        # Estrai i dati dallo stato
        speed_initial = state[0]        # Velocità all'inizio
        speed_final = state[122]        # Velocità alla fine dello stato
        delta_v = speed_final - speed_initial  # Differenza di velocità (delta V)

        throttle_buffer = state[2:62]   # Ultimi 60 valori di throttle
        brake_buffer = state[62:122]    # Ultimi 60 valori di brake
        
        # Valori attivi negli ultimi 20 secondi
        n_throttle_active = np.sum(throttle_buffer[-20:] > 0)
        n_brake_active = np.sum(brake_buffer[-20:] > 0)

        # Variabili base
        throttle_mean = 50
        brake_mean = 500
        covariance_matrix = BASE_COVARIANCE_MATRIX.copy()

        # Se delta_v < 0, incrementiamo leggermente la media del throttle per 20 step
        throttle_boost = 0
        if delta_v < 0:
            throttle_boost = 20

        # Se throttle attivi < brake attivi, favoriamo throttle > 0 per 20 step
        throttle_bias_boost = 0
        if n_throttle_active < n_brake_active:
            throttle_bias_boost = 20

        # Generazione dell'attack_list
        for i in range(500):
            # Aggressività progressiva
            throttle_mean_step = throttle_mean + (i / 20)  # Incrementa gradualmente ogni 25 step
            brake_mean_step = brake_mean - (i / 2)       # Riduce gradualmente il brake

            # Se throttle_boost è attivo
            if throttle_boost > 0:
                throttle_mean_step += 5  # Incremento temporaneo
                throttle_boost -= 1

            # Genera campioni
            sample = np.random.multivariate_normal(
                mean=[throttle_mean_step, brake_mean_step],
                cov=covariance_matrix
            )

            # Vincolo per favorire throttle se throttle_bias_boost è attivo
            if throttle_bias_boost > 0:
                if np.random.rand() < 0.7:  # 70% di probabilità di favorire throttle
                    sample[0] = max(sample[0], sample[1])
                throttle_bias_boost -= 1

            # Clipping ai range
            sample[0] = np.clip(sample[0], *THROTTLE_CLIP)  # Throttle
            sample[1] = np.clip(sample[1], *BRAKE_CLIP)     # Brake

            attack_list.append(sample)

        attack_list = np.array(attack_list)
        print("Nuova attack_list generata con dipendenze dallo stato iniziale.")

    # Seleziona il valore corrente
    throttle_candidate, brake_candidate = attack_list[current_index]
    current_index += 1

    # Vincolo: solo uno tra throttle e brake può essere attivo
    if throttle_candidate / 100 > brake_candidate / 1000:
        throttle = int(round(throttle_candidate))
        brake = 0
    else:
        throttle = 0
        brake = int(round(brake_candidate))

    return throttle, brake

# Esempio di utilizzo
if __name__ == "__main__":
    example_state = np.random.rand(20)  # Stato simulato
    throttle, brake = calculate_optimal_controls(example_state)
    print(f"Throttle: {throttle}, Brake: {brake}")'''