import pickle

def load_and_display_dataset(file_path):
    """
    Carica e visualizza il dataset salvato in formato .pkl.
    
    Args:
        file_path (str): Percorso al file del dataset.
    """
    try:
        with open(file_path, "rb") as f:
            dataset = pickle.load(f)

        print(f"Dataset caricato con successo! Numero di righe: {len(dataset)}\n")
        
        # Mostra i primi 5 esempi
        print("Esempi di dati dal dataset:")
        for idx, row in enumerate(dataset[:5]):
            print(f"\nRiga {idx + 1}:")
            print(f"  Input Throttle Mean: {row['input_throttle_mean']}")
            print(f"  Input Brake Mean: {row['input_brake_mean']}")
            print(f"  Max Speed: {row['max_speed']}")
            print(f"  Inputs (20 coppie): {row['inputs'][:2]} ...")  # Mostra solo le prime 2 coppie
            print(f"  Best Output (10 coppie): {row['best_output'][:2]} ...")  # Mostra solo le prime 2 coppie
            
    except FileNotFoundError:
        print(f"Errore: Il file {file_path} non esiste.")
    except Exception as e:
        print(f"Errore durante il caricamento del dataset: {e}")

if __name__ == "__main__":
    # Specifica il percorso del dataset
    dataset_path = "Attacker_dataset.pkl"
    
    # Carica e visualizza il dataset
    load_and_display_dataset(dataset_path)