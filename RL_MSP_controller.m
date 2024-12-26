function [throttle, brake] = RL_MSP_controller(EngineRPM, VehicleSpeed, time)
    % Controller chiamato ad ogni step di simulazione (1 secondo)
    % Mantiene i buffer degli ultimi 60 secondi di dati.
    % Le azioni (throttle, brake) restano casuali per tutta la durata.
    % Dopo 60 secondi, chiama Python per calcolare il cost_to_go dello stato attuale
    % (e internamente migliorare la policy), ma non influenza le azioni.

    persistent throttle_buffer brake_buffer speed_buffer rpm_buffer random_state initialized 
    persistent throttle_mean brake_mean covariance_matrix
    persistent cost_to_go_list

    coder.varsize('cost_to_go_list', [Inf, 2], [1, 0]); % Dichiara dimensioni variabili
    coder.extrinsic('py.RL_MSP_cost_to_go.cost_to_go');
    
    if isempty(initialized)
        % Inizializza i buffer vuoti
        throttle_buffer = [];
        brake_buffer = [];
        speed_buffer = [];
        rpm_buffer = [];
        
        % Parametri per i valori casuali iniziali
        throttle_mean = 50;
        brake_mean = 300;
        covariance_matrix = [100, -600; -600, 8000];

        % Seed casuale
        random_state = rng('shuffle'); % Imposta un seed casuale
        
        % Inizializza il cost_to_go_list con il contenuto del file, se esiste
        if isfile('.../MATLAB/TESI/RL_cost_to_go.mat')
            % Carica il file se esiste
            loaded_data = load('RL_cost_to_go.mat', 'cost_to_go_list');
            cost_to_go_list = loaded_data.cost_to_go_list;
        else
            % Inizializza cost_to_go_list come vuoto
            cost_to_go_list = zeros(0, 2); 
            % Crea il file e salva la variabile inizializzata
            save('RL_cost_to_go.mat', 'cost_to_go_list');
            fprintf('File RL_cost_to_go.mat non trovato. Creato nuovo file.\n');
        end
        
        initialized = true;
    end
    
    current_time = round(time);
    
    % Aggiorna i buffer speed e rpm
    speed_buffer = [speed_buffer; VehicleSpeed];
    rpm_buffer = [rpm_buffer; EngineRPM];
    if length(speed_buffer) > 60
        speed_buffer(1) = [];
        rpm_buffer(1) = [];
    end
    
    % Genera valori casuali prima dell'attacco con seed variabile
    rng(random_state.Seed + time, 'twister'); % Cambia il seed ogni volta
    
    % Generazione di campioni casuali
    inputs = mvnrnd([throttle_mean, brake_mean], covariance_matrix, 1);
    
    % Normalizzazione dei valori rispetto ai range massimi
    normalized_throttle = inputs(1) / 100; % throttle normalizzato su [0, 1]
    normalized_brake = inputs(2) / 1000;   % brake normalizzato su [0, 1]
    
    % Introduzione di un fattore casuale
    random_factor = rand; % Numero casuale tra 0 e 1
    
    % Logica per decidere chi impostare a zero
    if random_factor < 0.4
        % 40% delle volte imposta sempre il throttle
        throttle = max(0, min(100, round(normalized_throttle * 100))); % throttle limitato tra 0 e 100
        brake = 0;
    elseif random_factor > 0.6
        % 40% delle volte imposta sempre il brake
        throttle = 0;
        brake = max(0, min(1000, round(normalized_brake * 1000))); % brake limitato tra 0 e 1000
    else
        % 20% delle volte usa i valori normalizzati per decidere
        if normalized_throttle > normalized_brake
            % Usa throttle se maggiore
            throttle = max(0, min(100, round(normalized_throttle * 100)));
            brake = 0;
        else
            % Usa brake se maggiore
            throttle = 0;
            brake = max(0, min(1000, round(normalized_brake * 1000)));
        end
    end
    
    % Aggiorna i buffer throttle e brake
    throttle_buffer = [throttle_buffer; throttle];
    brake_buffer = [brake_buffer; brake];
    if length(throttle_buffer) > 60
        throttle_buffer(1) = [];
        brake_buffer(1) = [];
    end
    
    % Dopo i 60 secondi, chiama Python per ottenere il cost_to_go dello stato attuale
    if current_time > 60
        if length(throttle_buffer) == 60 && length(brake_buffer) == 60
            Speed_1_min_ago = speed_buffer(1);
            RPM_1_min_ago = rpm_buffer(1);
            last_60_throttle = throttle_buffer(:)';
            last_60_brake = brake_buffer(:)';
            Speed_now = speed_buffer(end);
            RPM_now = rpm_buffer(end);

            state_vector = [Speed_1_min_ago, RPM_1_min_ago, ...
                            last_60_throttle, last_60_brake, ...
                            Speed_now, RPM_now];

            % Chiama la funzione Python che calcola il cost_to_go (e internamente può migliorare la policy)
            cost_to_go = py.RL_MSP_cost_to_go.cost_to_go(state_vector);
            
            cost_to_go = double(cost_to_go);  % Converte da Python a double
            
            % Ora cost_to_go contiene il valore stimato del costo per violare le specifiche dallo stato attuale.
            % Salva in cost_to_go_list
            cost_to_go_list = [cost_to_go_list; current_time, cost_to_go];
            
            % Salva l'aggiornamento su file
            save('RL_cost_to_go.mat', 'cost_to_go_list');
        end
    end
end