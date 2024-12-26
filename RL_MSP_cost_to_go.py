import numpy as np
from tensorflow.keras.models import load_model

# Carica il modello del cost-to-go
model = load_model('cost_predictor_model_R.h5')

def normalize_state(state_vector):
    # state_vector layout:
    # 0: Speed_1_min_ago (0-130)
    # 1: RPM_1_min_ago (0-4600)
    # 2-61: Throttle (60 valori, range 0-100)
    # 62-121: Brake (60 valori, range 0-2500)
    # 122: Speed_now (0-130)
    # 123: RPM_now (0-4600)
    #
    # Applichiamo la stessa normalizzazione usata in training:
    #
    # Speed: /130
    # RPM: /4600
    # Throttle: /100
    # Brake: /2500
    #
    # Non dimenticare: state_vector potrebbe avere dimensioni diverse a seconda di come l'hai impostato,
    # assicurati che gli indici corrispondano correttamente.

    state_norm = state_vector.copy().astype('float32')

    # Speed_1_min_ago
    state_norm[0] = state_norm[0] / 120.0
    # RPM_1_min_ago
    state_norm[1] = state_norm[1] / 4500.0

    # Throttle (colonne 2 a 61 inclusi)
    state_norm[2:62] = state_norm[2:62] / 100.0

    # Brake (colonne 62 a 121 inclusi)
    state_norm[62:122] = state_norm[62:122] / 1000.0

    # Speed_now
    state_norm[122] = state_norm[122] / 120.0
    # RPM_now
    state_norm[123] = state_norm[123] / 4500.0

    return state_norm

def cost_to_go(state_vector):
    # Assicuriamoci che state_vector sia un array numpy:
    if not isinstance(state_vector, np.ndarray):
        state_vector = np.array(state_vector, dtype=np.float32)

    # Normalizza l'input
    state_norm = normalize_state(state_vector)

    # Ridimensiona l'input per il modello: (1, num_features)
    state_norm = state_norm.reshape(1, -1)

    # Predici il cost_to_go normalizzato
    cost_to_go_norm = model.predict(state_norm, verbose=0)  # questo è normalizzato tra 0 e 1 se hai addestrato così

    # cost_to_go_norm è un array Nx1, noi vogliamo un singolo valore
    cost_to_go_norm_value = float(cost_to_go_norm[0,0])

    # Denormalizzazione del cost_to_go
    # Durante il training il costo era normalizzato dividendo per 100, quindi qui moltiplichiamo per 100
    cost_value = cost_to_go_norm_value * 200.0

    return cost_value