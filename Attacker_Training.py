import torch
import torch.nn as nn
import torch.optim as optim
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
# Funzione di perdita personalizzata
# ------------------------------
def custom_loss(predicted, state, throttle_max=100, brake_min=0):
    """
    Calcola la loss totale:
    - Penalità per throttle lontano dal massimo.
    - Penalità per brake lontano dal minimo.
    - Penalità per distanza dalla media dei valori positivi nello stato.
    - Penalità per throttle e brake entrambi uguali a zero.
    - Penalità per numero di throttle > 0 nelle predizioni rispetto al valore scalato dallo stato.
    """
    # Estrai throttle e brake dallo stato
    throttle_state = state[:, :, 0]
    brake_state = state[:, :, 1]

    # Calcola la media di throttle e brake solo per valori > 0
    throttle_mean = torch.sum(throttle_state * (throttle_state > 0), dim=1, keepdim=True) / \
                    torch.sum((throttle_state > 0).float(), dim=1, keepdim=True).clamp(min=1)
    brake_mean = torch.sum(brake_state * (brake_state > 0), dim=1, keepdim=True) / \
                 torch.sum((brake_state > 0).float(), dim=1, keepdim=True).clamp(min=1)

    # Maschere logiche per le condizioni specifiche
    throttle_mask = (predicted[:, :, 0] > 0) & (predicted[:, :, 1] == 0)  # Throttle > 0 e Brake == 0
    brake_mask = (predicted[:, :, 1] > 0) & (predicted[:, :, 0] == 0)     # Brake > 0 e Throttle == 0

    # Penalità per distanza dalla media con condizioni applicate
    throttle_mean_loss = torch.mean(((predicted[:, :, 0] - throttle_mean) ** 2) * throttle_mask)
    brake_mean_loss = torch.mean(((predicted[:, :, 1] - brake_mean) ** 2) * brake_mask) / 10

    # Penalità per throttle e brake entrambi a zero
    both_zero_penalty = torch.mean(((predicted[:, :, 0] == 0) & (predicted[:, :, 1] == 0)).float())

    # Penalità per throttle lontano dal massimo
    throttle_loss = torch.mean((throttle_max - predicted[:, :, 0]) ** 2)

    # Penalità per brake lontano dal minimo
    brake_loss = torch.mean((predicted[:, :, 1] - brake_min) ** 2) / 10

    # ---------------------------
    # Loss Totale
    # ---------------------------
    total_loss = (
        0.01 * throttle_loss + 
        0.01 * brake_loss + 
        0.05 * throttle_mean_loss + 
        0.05 * brake_mean_loss + 
        1000 * both_zero_penalty
    )

    return total_loss

# ------------------------------
# Generazione dati di esempio
# ------------------------------
def generate_dummy_data(batch_size=64, seq_len=60):
    """
    Genera 60 coppie (throttle, brake) per batch usando una distribuzione multivariata
    e logica per garantire che solo uno dei due valori sia > 0.
    """
    state = torch.zeros(batch_size, seq_len, 2)  # Inizializza il tensore con zeri
    
    for b in range(batch_size):
        # Genera medie casuali per throttle e brake
        throttle_mean = np.random.uniform(40, 100)  # Media casuale tra 40 e 85
        brake_mean = np.random.uniform(0, 400)  # Media casuale tra 400 e 1000
        
        # Matrice di covarianza negativa
        covariance_matrix = [[100, -600],
                             [-600, 8000]]

        # Genera campioni casuali usando la distribuzione multivariata
        samples = np.random.multivariate_normal([throttle_mean, brake_mean], covariance_matrix, seq_len)

        for t in range(seq_len):
            # Normalizza i valori rispetto ai range massimi
            normalized_throttle = max(0, min(1, samples[t, 0] / 100))  # throttle normalizzato su [0, 1]
            normalized_brake = max(0, min(1, samples[t, 1] / 1000))    # brake normalizzato su [0, 1]

            # Fattore casuale per decidere chi impostare a zero
            random_factor = np.random.rand()
            if random_factor < 0.4:
                # 40%: throttle > 0, brake = 0
                throttle = max(0, min(100, round(normalized_throttle * 100)))
                brake = 0
            elif random_factor > 0.6:
                # 40%: brake > 0, throttle = 0
                throttle = 0
                brake = max(0, min(1000, round(normalized_brake * 1000)))
            else:
                # 20%: usa i valori normalizzati per decidere
                if normalized_throttle > normalized_brake:
                    throttle = max(0, min(100, round(normalized_throttle * 100)))
                    brake = 0
                else:
                    throttle = 0
                    brake = max(0, min(1000, round(normalized_brake * 1000)))

            # Assegna i valori a `state`
            state[b, t, 0] = throttle
            state[b, t, 1] = brake

    return state

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
    adjusted = torch.stack([adjusted_throttle, adjusted_brake], dim=2)

    return adjusted


# ------------------------------
# Funzione di addestramento
# ------------------------------
def train_model(model, epochs=500, batch_size=64, seq_len=60):
    """
    Allenamento del modello.
    Args:
        model: Modello LSTM da allenare.
        epochs: Numero di epoche.
        batch_size: Dimensione del batch.
        seq_len: Lunghezza della sequenza di input.
    """
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    for epoch in range(epochs):
        model.train()
        state = generate_dummy_data(batch_size, seq_len)  # Genera dati di input con 60 coppie
        optimizer.zero_grad()

        # Genera predizioni e applica enforce_exclusivity
        predicted = model(state)
        predicted_adjusted = enforce_exclusivity(predicted)

        # Calcola la loss
        total_loss = custom_loss(predicted_adjusted, state)

        # Backpropagation e ottimizzazione
        total_loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}")

# ------------------------------
# Addestramento del Modello
# ------------------------------
input_dim = 2  # Throttle e brake
hidden_dim = 64
output_dim = 2  # Throttle e brake predetti
num_layers = 2

# Crea il modello
model = AttackPredictor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

# Avvia l'addestramento
train_model(model)

# Salva il modello addestrato
torch.save(model.state_dict(), "AttackPredictor.pth")
print("Modello addestrato e salvato come 'AttackPredictor.pth'")


