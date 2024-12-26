import numpy as np
import nevergrad as ng
import pickle
import matlab.engine

def train_model(output_model_path="Attacker_optimized_model.pkl", budget=5000):
    """
    Addestra un modello usando Nevergrad per massimizzare la velocità.
    Salva il modello addestrato in un file.
    Args:
        output_model_path (str): Percorso per salvare il modello ottimizzato.
        budget (int): Numero massimo di iterazioni per Nevergrad.
    """
    
    # Parametri target
    mean = [50, 250]  # Media target
    covariance_matrix = [[100, 0], [0, 5000]]  # Covarianza target

    # Funzione obiettivo per l'ottimizzazione
    def objective(new_pairs_flat, iteration):
        """
        Funzione obiettivo per massimizzare la velocità simulata.
        Args:
            new_pairs_flat (numpy.ndarray): Array piatta [1200] (600 coppie di throttle e brake).
            iteration (int): Numero dell'iterazione corrente.
        Returns:
            float: Penalità negativa basata sull'aumento di velocità.
        """
        try:
            # Reshape per ottenere la forma (600, 2)
            new_pairs = np.array(new_pairs_flat).reshape(600, 2)
        except ValueError:
            raise ValueError(f"L'input non può essere convertito in una matrice (600, 2): {new_pairs_flat}")
    
        # Calcola la penalità per media e varianza in ogni blocco di 10 coppie
        total_penalty = 0
        for i in range(0, 600, 10):
            block = new_pairs[i:i + 10]
            
            # Calcola media e varianza per il blocco
            throttle_mean = np.mean(block[:, 0])
            brake_mean = np.mean(block[:, 1])
            throttle_var = np.var(block[:, 0])
            brake_var = np.var(block[:, 1])
    
            # Penalità per deviazione dalla media e varianza target
            mean_penalty = (
                abs(throttle_mean - mean[0]) + abs(brake_mean - mean[1])  # Deviazione dalle medie
            )
            var_penalty = (
                abs(throttle_var - covariance_matrix[0][0]) +
                abs(brake_var - covariance_matrix[1][1])  # Deviazione dalle varianze
            )
    
            # Penalità totale per il blocco
            total_penalty += mean_penalty + var_penalty
    
        # Simula il guadagno di velocità per tutte le 600 coppie
        speed_gain_total = simulate_speed(new_pairs)
        
        # Normalizza e bilancia le penalità
        speed_gain_normalized = speed_gain_total / 2 # Normalizza il guadagno di velocità
        penalty_normalized = total_penalty / 1000  # Normalizza le penalità (aggiusta in base ai valori osservati)
    
        # Funzione obiettivo bilanciata
        loss = -speed_gain_normalized + penalty_normalized
        
        # Feedback sulla simulazione corrente
        print(f"Iterazione: {iteration + 1}")
        print(f"Guadagno di velocità totale: {speed_gain_total} (normalizzato: {speed_gain_normalized})")
        print(f"Penalità totale: {total_penalty} (normalizzato: {penalty_normalized})")
        print(f"Valore della funzione obiettivo: {loss}")
        
        return loss

    # Genera 600 coppie iniziali che rispettano media, varianza e limiti
    initial_values = np.random.multivariate_normal(mean, covariance_matrix, size=600).flatten()
    initial_values = np.clip(initial_values, np.tile([0, 0], 600), np.tile([100, 1000], 600))

    # Configura l'ottimizzatore Nevergrad
    instrumentation = ng.p.Array(init=initial_values)
    instrumentation.set_bounds(lower=np.zeros(1200), upper=np.tile([100, 1000], 600))
    
    # Sostituisci OnePlusOne con CMA
    optimizer = ng.optimizers.CMA(parametrization=instrumentation, budget=budget, num_workers=50)
    
    # Esegui l'ottimizzazione con feedback sulle iterazioni
    print("Inizio ottimizzazione con CMA...")
    best_value = float('inf')  # Inizializza il miglior valore (per minimizzazione)
    
    for iteration in range(budget):
        recommendation = optimizer.ask()
        value = objective(recommendation.value, iteration)
        optimizer.tell(recommendation, value)
    
        # Aggiorna il miglior valore se necessario
        if value < best_value:
            best_value = value
    
        # Feedback ogni 50 iterazioni
        if (iteration + 1) % 50 == 0:
            print(f"Iterazione {iteration + 1}/{budget}: Miglior valore attuale: {-best_value}")  # Negativo per il guadagno positivo

    # Ottieni la migliore raccomandazione
    recommendation = optimizer.provide_recommendation()

    # Salva il modello ottimizzato
    optimized_pairs = recommendation.value.reshape(600, 2)
    with open(output_model_path, "wb") as f:
        pickle.dump(optimized_pairs, f)

    print(f"Modello ottimizzato salvato in: {output_model_path}")


def simulate_speed(new_pairs):
    """
    Simula il guadagno di velocità basato sulle coppie throttle e brake
    utilizzando la funzione MATLAB `Attacker_simulate_speed`.
    Args:
        new_pairs (numpy.ndarray): Coppie di throttle e brake [600, 2].
    Returns:
        float: Guadagno simulato di velocità.
    """
    if new_pairs.shape != (600, 2):
        raise ValueError("Le coppie devono avere dimensione [600, 2]")

    # Avvia MATLAB Engine
    eng = matlab.engine.start_matlab()

    try:
        # Converti le coppie in array MATLAB
        throttle_input = matlab.double(new_pairs[:, 0].tolist())
        brake_input = matlab.double(new_pairs[:, 1].tolist())

        # Chiama la funzione MATLAB con tutte le 600 coppie
        speed_gain = eng.Attacker_simulate_speed(throttle_input, brake_input)

    except Exception as e:
        print(f"Errore durante la simulazione: {e}")
        speed_gain = -1e6  # Penalità in caso di errore

    finally:
        # Chiudi MATLAB Engine
        eng.quit()

    return speed_gain


if __name__ == "__main__":
    # Percorso per salvare il modello ottimizzato
    output_model_path = "Attacker_optimized_model.pkl"

    # Numero di iterazioni per l'ottimizzatore
    budget = 5000

    print("Inizio addestramento del modello usando Nevergrad...")
    
    # Avvio del processo di addestramento
    train_model(output_model_path=output_model_path, budget=budget)

    print(f"Addestramento completato. Modello salvato in: {output_model_path}")