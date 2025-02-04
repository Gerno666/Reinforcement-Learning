import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import psutil
import matplotlib.pyplot as plt

# Custom callback to monitor resources
class ResourceMonitor(Callback):
    def __init__(self):
        self.cpu_usage = []
        self.ram_usage = []
        self.timestamps = []
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Record resource usage
        self.cpu_usage.append(psutil.cpu_percent())
        self.ram_usage.append(psutil.virtual_memory().used / (1024 ** 2))  # RAM in MB
        self.timestamps.append(time.time() - self.start_time)

    def plot_resources(self):
        # Plot CPU and RAM usage over time
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(self.timestamps, self.cpu_usage, label="CPU Usage (%)", color="blue")
        plt.xlabel("Time (s)")
        plt.ylabel("CPU Usage (%)")
        plt.title("CPU Usage During Training")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.ram_usage, label="RAM Usage (MB)", color="green")
        plt.xlabel("Time (s)")
        plt.ylabel("RAM Usage (MB)")
        plt.title("RAM Usage During Training")
        plt.legend()

        plt.tight_layout()
        plt.savefig('CPU_RAM_training_policy.png')  # Save the figure
        plt.show()

# Instantiate the resource monitor
monitor = ResourceMonitor()

# Carica i dati
file_path = 'RL_10k_simulations_data.csv'  # Percorso del file CSV
data = pd.read_csv(file_path)

# Dividi i dati in input (X) e target (y)
X = data.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
y = data.iloc[:, -1].values   # Ultima colonna (costo)

# Verifica NaN nei dati
print(f"NaN nei dati di input: {np.isnan(X).sum()}")
print(f"NaN nei target: {np.isnan(y).sum()}")

# Rimuovi eventuali righe con NaN nel target
mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

X = X.astype(np.float32)  # Converti tutti i valori in float
y = y.astype(np.float32)  # Converti tutti i valori in float

# Normalizzazione manuale:
# Colonna 0: Speed_iniziale [0,130]
X[:, 0] = X[:, 0] / 120.0

# Colonna 1: RPM_iniziale [0,4600]
X[:, 1] = X[:, 1] / 4500.0

# Colonne 2-61: Throttle (60 valori, range 0-100)
X[:, 2:62] = X[:, 2:62] / 100.0

# Colonne 62-121: Brake (60 valori, range 0-2500)
X[:, 62:122] = X[:, 62:122] / 1000.0

# Colonna 122: Speed_finale [0,130]
X[:, 122] = X[:, 122] / 120.0

# Colonna 123: RPM_finale [0,4600]
X[:, 123] = X[:, 123] / 4500.0

# Target (costo) [0,100]
y = y / 100.0

# Salva il dataset normalizzato
normalized_data = np.hstack([X, y.reshape(-1, 1)])  # Combina X e y normalizzati
normalized_columns = data.columns  # Usa i nomi delle colonne originali
normalized_df = pd.DataFrame(normalized_data, columns=normalized_columns)
normalized_df.to_csv('RL_10k_simulations_data_normalized_test.csv', index=False)
print("Dataset normalizzato salvato in RL_10k_simulations_data_normalized.csv")

# Split dei dati in training e test (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruzione del modello
model = Sequential()

# Primo layer: input
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Primo layer nascosto
model.add(BatchNormalization())  # Batch Normalization
model.add(Dropout(0.3))  # Dropout

# Secondo layer nascosto
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Terzo layer nascosto
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Layer di output (un singolo valore per il costo)
model.add(Dense(1)) 

# Ottimizzazione con Adam e learning rate iniziale
optimizer = Adam(learning_rate=0.0001)

# Compilazione del modello
model.compile(loss='mean_squared_error', optimizer=optimizer)


# Addestramento del modello
history = model.fit(X_train, y_train, epochs=200, batch_size=64, 
                    validation_data=(X_test, y_test),
                    verbose=1, callbacks=[monitor])

# Valutazione del modello sui dati di test
loss = model.evaluate(X_test, y_test)
print(f'Errore quadratico medio (MSE) sui dati di test: {loss}')

# Salva il modello addestrato
model.save('cost_predictor_model_test.h5')
print('Modello addestrato e salvato come cost_predictor_model.h5')

# Previsioni con il modello
y_pred = model.predict(X_test)

# Convertiamo le previsioni e il target di test allo spazio originale (denormalizzando)
y_pred_original = y_pred * 100.0
y_test_original = y_test * 100.0

# Stampa alcune previsioni per il debug
print(f"Prime 10 previsioni dei costi (denormalizzate): {y_pred_original[:10].flatten()}")
print(f"Primi 10 costi reali (denormalizzati): {y_test_original[:10].flatten()}")

# Salva le previsioni e i valori reali in un file CSV
predictions_df = pd.DataFrame({
    "Actual Cost": y_test_original.flatten(),
    "Predicted Cost": y_pred_original.flatten()
})
#predictions_df.to_csv('RL_10k_predictions.csv', index=False)
print("Previsioni e valori reali salvati in RL_10k_predictions.csv")

# Plot resource usage
monitor.plot_resources()


'''import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carica i dati
file_path = 'RL_10k_simulations_data.csv'  # Percorso del file CSV
data = pd.read_csv(file_path)

# Dividi i dati in input (X) e target (y)
X = data.iloc[:, :-1].values  # Tutte le colonne tranne l'ultima
y = data.iloc[:, -1].values   # Ultima colonna (costo)

# Verifica NaN nei dati
print(f"NaN nei dati di input: {np.isnan(X).sum()}")
print(f"NaN nei target: {np.isnan(y).sum()}")

# Rimuovi eventuali righe con NaN nel target
mask = ~np.isnan(y)
X = X[mask]
y = y[mask]

X = X.astype(np.float32)  # Converti tutti i valori in float
y = y.astype(np.float32)  # Converti tutti i valori in float

# Normalizzazione manuale:
# Colonna 0: Speed_iniziale [0,130]
X[:, 0] = X[:, 0] / 120.0

# Colonna 1: RPM_iniziale [0,4600]
X[:, 1] = X[:, 1] / 4500.0

# Colonne 2-61: Throttle (60 valori, range 0-100)
X[:, 2:62] = X[:, 2:62] / 100.0

# Colonne 62-121: Brake (60 valori, range 0-2500)
X[:, 62:122] = X[:, 62:122] / 1000.0

# Colonna 122: Speed_finale [0,130]
X[:, 122] = X[:, 122] / 120.0

# Colonna 123: RPM_finale [0,4600]
X[:, 123] = X[:, 123] / 4500.0

# Target (costo) [0,100]
y = y / 100.0

# Salva il dataset normalizzato
normalized_data = np.hstack([X, y.reshape(-1, 1)])  # Combina X e y normalizzati
normalized_columns = data.columns  # Usa i nomi delle colonne originali
normalized_df = pd.DataFrame(normalized_data, columns=normalized_columns)
normalized_df.to_csv('RL_10k_simulations_data_normalized.csv', index=False)
print("Dataset normalizzato salvato in RL_10k_simulations_data_normalized.csv")

# Split dei dati in training e test (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Costruzione del modello
model = Sequential()

# Primo layer: input
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Primo layer nascosto
model.add(BatchNormalization())  # Batch Normalization
model.add(Dropout(0.3))  # Dropout

# Secondo layer nascosto
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Terzo layer nascosto
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

# Layer di output (un singolo valore per il costo)
model.add(Dense(1)) 

# Ottimizzazione con Adam e learning rate iniziale
optimizer = Adam(learning_rate=0.0001)

# Compilazione del modello
model.compile(loss='mean_squared_error', optimizer=optimizer)


# Addestramento del modello
history = model.fit(X_train, y_train, epochs=200, batch_size=64, 
                    validation_data=(X_test, y_test),
                    verbose=1)

# Valutazione del modello sui dati di test
loss = model.evaluate(X_test, y_test)
print(f'Errore quadratico medio (MSE) sui dati di test: {loss}')

# Salva il modello addestrato
model.save('cost_predictor_model_test.h5')
print('Modello addestrato e salvato come cost_predictor_model.h5')

# Previsioni con il modello
y_pred = model.predict(X_test)

# Convertiamo le previsioni e il target di test allo spazio originale (denormalizzando)
y_pred_original = y_pred * 100.0
y_test_original = y_test * 100.0

# Stampa alcune previsioni per il debug
print(f"Prime 10 previsioni dei costi (denormalizzate): {y_pred_original[:10].flatten()}")
print(f"Primi 10 costi reali (denormalizzati): {y_test_original[:10].flatten()}")

# Salva le previsioni e i valori reali in un file CSV
predictions_df = pd.DataFrame({
    "Actual Cost": y_test_original.flatten(),
    "Predicted Cost": y_pred_original.flatten()
})
predictions_df.to_csv('RL_10k_predictions.csv', index=False)
print("Previsioni e valori reali salvati in RL_10k_predictions.csv")'''