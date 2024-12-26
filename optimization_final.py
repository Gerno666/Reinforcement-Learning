import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pickle
import os

# Disable detailed TensorFlow logs
tf.get_logger().setLevel('ERROR')

# Path to save and load the optimized model weights
MODEL_WEIGHTS_FILE = "optimized_model_weights.pkl"

def create_model():
    """
    Create a neural network model with 3 layers, updated for 20 input features.
    """
    model = Sequential([
        Dense(10, input_shape=(20,), activation='relu'),  # First hidden layer with 10 units
        Dense(5, activation='relu'),                      # Second hidden layer with 5 units
        Dense(2)  # Output layer with 2 neurons for throttle and brake
    ])
    return model

def optimize_neural_network_weights(single_state):
    """
    Optimize the weights of the neural network using Nevergrad based on a single state
    and save the optimized weights for future use.
    """
    import nevergrad as ng
    import matlab.engine
    import gc

    # Start the MATLAB engine
    eng = matlab.engine.start_matlab()

    # Create the model
    model = create_model()
    
    # Get the total number of weights the model expects
    expected_weights = model.get_weights()
    total_weights = sum(w.size for w in expected_weights)
    print(f"Number of expected weights in the model: {total_weights}")

    # Set up Nevergrad optimizer with CMA algorithm
    optimizer = ng.optimizers.CMA(parametrization=total_weights, budget=2000, num_workers=1)

    # Counter for iterations
    iteration_count = 0

    def objective(weights):
        nonlocal iteration_count
        iteration_count += 1  # Increment the iteration counter

        # Set the weights in the model
        reshaped_weights = []
        start = 0
        for w in expected_weights:
            size = w.size
            reshaped_weights.append(np.array(weights[start:start + size]).reshape(w.shape))
            start += size
        model.set_weights(reshaped_weights)

        # Convert the state to numpy array
        state_np = np.array(single_state).reshape(1, -1)

        # Predict throttle and brake using the current weights
        prediction = model.predict(state_np)
        throttle = max(0, min(int(round(prediction[0, 0])), 100))
        brake = max(0, min(int(round(prediction[0, 1])), 2500))

        # Simulate and calculate violation
        try:
            average_violation = eng.run_simulation_in_simulink(throttle, brake)
            if average_violation is not None:
                print(f"Iteration {iteration_count}: Throttle={throttle}, Brake={brake}, Violation={average_violation}")
                return -average_violation  # Return negative to minimize violation
        except Exception as e:
            print(f"Simulation failed during iteration {iteration_count}: {e}")
            return 0  # Neutral value for failures

        return 0  # Fallback in case of error

    try:
        # Run the optimizer to find the best weights
        recommendation = optimizer.minimize(objective)
        best_weights = recommendation.value

        # Save the optimized weights to a file
        with open(MODEL_WEIGHTS_FILE, "wb") as f:
            pickle.dump(best_weights, f)

        print(f"Optimized weights saved after {iteration_count} iterations.")
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
    finally:
        # Shut down the MATLAB engine
        eng.quit()
        gc.collect()

def calculate_optimal_controls(state):
    """
    Predict optimal throttle and brake values using the optimized neural network model.
    If weights are not available, optimize them using Nevergrad.
    """
    # Check if the weights file exists
    if not os.path.exists(MODEL_WEIGHTS_FILE):
        print("Optimized weights not found. Running optimization...")
        # Usa lo stato passato da MATLAB come punto di partenza per l'ottimizzazione
        optimize_neural_network_weights(state)

    # Create the model
    model = create_model()

    # Load the optimized weights
    with open(MODEL_WEIGHTS_FILE, "rb") as f:
        best_weights = pickle.load(f)

    # Set the optimized weights to the model
    reshaped_weights = []
    start = 0
    expected_weights = model.get_weights()
    for w in expected_weights:
        size = w.size
        reshaped_weights.append(np.array(best_weights[start:start + size]).reshape(w.shape))
        start += size
    model.set_weights(reshaped_weights)

    # Predict throttle and brake
    state_np = np.array(state).reshape(1, -1)
    prediction = model.predict(state_np)
    throttle = max(0, min(int(round(prediction[0, 0])), 100))
    brake = max(0, min(int(round(prediction[0, 1])), 2500))

    return throttle, brake