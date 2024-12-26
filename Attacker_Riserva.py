import torch
import torch.nn as nn
import numpy as np

# ------------------------------
# Modello LSTM
# ------------------------------
class AttackPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=2, num_layers=2):
        super(AttackPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # Output dimensione 2 (throttle, brake)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -10:, :])  # Prendi le ultime 10 previsioni
        return output

# ------------------------------
# Carica il modello
# ------------------------------
def load_trained_model(model_path="AttackPredictor.pth"):
    model = AttackPredictor(input_dim=2, hidden_dim=64, output_dim=2, num_layers=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# ------------------------------
# Generazione di 60 coppie (throttle, brake)
# ------------------------------
def generate_60_pairs(seed=None):
    """
    Genera 60 coppie (throttle, brake) utilizzando la media e covarianza definite,
    con l'introduzione di una logica per impostare throttle o brake a zero.
    """
    if seed is not None:
        np.random.seed(seed)  # Imposta il seed per la riproducibilità

    # Definizione della media e della matrice di covarianza negativa
    throttle_mean = 50
    brake_mean = 300
    covariance_matrix = [[200, -600], [-600, 2000]]

    throttle_brake_pairs = []

    for _ in range(60):  # Genera 60 coppie
        # Genera campioni casuali dalla distribuzione multivariata
        inputs = np.random.multivariate_normal([throttle_mean, brake_mean], covariance_matrix)

        # Normalizzazione
        normalized_throttle = inputs[0] / 100  # Normalizza tra [0, 1]
        normalized_brake = inputs[1] / 1000    # Normalizza tra [0, 1]

        # Introduzione della logica
        random_factor = np.random.rand()

        if random_factor < 0.4:
            throttle = max(0, min(100, round(normalized_throttle * 100)))
            brake = 0
        elif random_factor > 0.6:
            throttle = 0
            brake = max(0, min(1000, round(normalized_brake * 1000)))
        else:
            if normalized_throttle > normalized_brake:
                throttle = max(0, min(100, round(normalized_throttle * 100)))
                brake = 0
            else:
                throttle = 0
                brake = max(0, min(1000, round(normalized_brake * 1000)))

        throttle_brake_pairs.append([throttle, brake])

    # Converte in tensore torch
    state = torch.tensor(throttle_brake_pairs, dtype=torch.float32).unsqueeze(0)  # Shape: [1, 60, 2]
    return state


# ------------------------------
# Test del modello
# ------------------------------

def enforce_exclusivity(predicted):
    """
    Forza che solo un valore tra throttle e brake sia > 0 con varianza applicata:
    - Introduce una varianza positiva e negativa ai valori > 0.
    - 40%: throttle > 0, brake = 0.
    - 40%: brake > 0, throttle = 0.
    - 20%: confronto tra throttle e brake/10.
    I valori negativi vengono forzati a 0 e limitati ai range:
    - Throttle: [0, 100]
    - Brake: [0, 1000]
    """
    # Genera un fattore casuale per ciascun elemento della batch
    random_factor = torch.rand(predicted.shape[:2], device=predicted.device)  # Shape: [batch_size, seq_len]

    # Estrai throttle e brake dalle predizioni e clampa i valori nei range consentiti
    throttle = torch.clamp(predicted[:, :, 0], min=0.0, max=100.0)  # Clampa throttle tra 0 e 100
    brake = torch.clamp(predicted[:, :, 1], min=0.0, max=1000.0)    # Clampa brake tra 0 e 1000

    # Introduci la varianza ai valori di throttle > 0
    positive_throttle_mask = throttle > 0
    throttle_variance = torch.randn_like(throttle) * 15
    throttle[positive_throttle_mask] += throttle_variance[positive_throttle_mask]

    # Introduci la varianza ai valori di brake > 0
    positive_brake_mask = brake > 0
    brake_variance = torch.randn_like(brake) * 60
    brake[positive_brake_mask] += brake_variance[positive_brake_mask]

    # Riclampare i valori dopo aver aggiunto varianza
    throttle = torch.clamp(throttle, min=0.0, max=100.0)
    brake = torch.clamp(brake, min=0.0, max=1000.0)

    # Crea tensori vuoti per i risultati
    adjusted_throttle = torch.zeros_like(throttle)
    adjusted_brake = torch.zeros_like(brake)

    # Logica per ciascun caso
    # Caso 1: random < 0.4 -> throttle > 0, brake = 0
    throttle_only_mask = random_factor < 0.4
    adjusted_throttle[throttle_only_mask] = throttle[throttle_only_mask]
    adjusted_brake[throttle_only_mask] = 0

    # Caso 2: random > 0.6 -> brake > 0, throttle = 0
    brake_only_mask = random_factor > 0.6
    adjusted_throttle[brake_only_mask] = 0
    adjusted_brake[brake_only_mask] = brake[brake_only_mask]

    # Caso 3: 0.4 <= random <= 0.6 -> confronto throttle e brake / 10
    compare_mask = (random_factor >= 0.4) & (random_factor <= 0.6)
    throttle_condition = throttle > (brake / 10)

    # Se la condizione è vera, throttle > 0
    adjusted_throttle[compare_mask & throttle_condition] = throttle[compare_mask & throttle_condition]
    adjusted_brake[compare_mask & throttle_condition] = 0

    # Se la condizione è falsa, brake > 0
    adjusted_brake[compare_mask & ~throttle_condition] = brake[compare_mask & ~throttle_condition]
    adjusted_throttle[compare_mask & ~throttle_condition] = 0

    # Ricombina throttle e brake in un tensore finale
    adjusted = torch.stack([adjusted_throttle, adjusted_brake], dim=2).int()

    return adjusted


def test_model(model):
    """
    Testa il modello su uno stato generato di 60 coppie e predice 10 coppie di output.
    """
    # Genera 60 coppie di input
    input_state = generate_60_pairs(seed=42)  # Imposta un seed per la riproducibilità
    print("Input State (60 coppie di throttle e brake):")
    print(input_state)  # Rimuovo squeeze per mantenere il formato corretto

    # Predizione
    with torch.no_grad():
        input_tensor = torch.tensor(input_state, dtype=torch.float32)  # Input già in forma corretta
        if len(input_tensor.shape) == 2:  # Aggiungi la dimensione batch se manca
            input_tensor = input_tensor.unsqueeze(0)  # [1, 60, 2]
        
        predictions = model(input_tensor)  # Predice le coppie throttle, brake
        predictions_adjusted = enforce_exclusivity(predictions)  # Applica la condizione di esclusività

    print("\nPredicted Output (10 coppie di throttle e brake):")
    print(predictions_adjusted.squeeze(0).numpy())


def load_trained_model(model_path="AttackPredictor.pth", input_dim=2, hidden_dim=64, output_dim=2, num_layers=2):
    """
    Carica il modello LSTM addestrato.
    """
    model = AttackPredictor(input_dim, hidden_dim, output_dim, num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def attack(input_state):
    """
    Testa il modello su uno stato passato di 60 coppie e predice 10 coppie di output.
    Args:
        input_state (numpy.ndarray): Stato di input di dimensione [60, 2].
        model_path (str): Percorso del file contenente il modello LSTM.
    """

    # Carica il modello automaticamente
    model_path="AttackPredictor.pth"
    model = load_trained_model(model_path)

    # Predizione del modello
    with torch.no_grad():
        input_tensor = torch.tensor(input_state, dtype=torch.float32)  # Converte in tensore
        if len(input_tensor.shape) == 2:  # Aggiunge la dimensione batch se manca
            input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, 60, 2]

        # Passa lo stato attraverso il modello
        predictions = model(input_tensor)  # Output: [1, 10, 2]
        predictions_adjusted = enforce_exclusivity(predictions)  # Applica la condizione di esclusività

    return (predictions_adjusted.squeeze(0).numpy())  # Rimuove la dimensione batch e converte in numpy


# ------------------------------
# Esecuzione
# ------------------------------
if __name__ == "__main__":
    # Carica il modello addestrato
    model = load_trained_model("AttackPredictor.pth")
    
    # Testa il modello
    test_model(model)


