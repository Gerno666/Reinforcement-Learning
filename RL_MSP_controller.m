%{
function [throttle, brake, terminate] = RL_MSP_controller(EngineRPM, VehicleSpeed, time)
    % Inizializza le variabili di output
    throttle = 0;
    brake = 0;
    terminate = false; % Segnale per terminare la simulazione

    % Parametri globali
    persistent sliding_window stop_time initialized max_window_size attack_started attack_time saved_window
    persistent predicted_pairs pair_index simulation_id cost_to_go_list

    % Dichiarazione estrinseca della funzione Python
    coder.extrinsic('py.numpy.array');
    coder.extrinsic('py.Attacker.attack');
    coder.extrinsic('py.RL_MSP_cost_to_go.cost_to_go');
    coder.varsize('cost_to_go_list', [Inf, 2], [1, 0]); % Dichiara dimensioni variabili

    % Inizializza variabili alla prima chiamata
    if isempty(initialized)
        % Dimensione della sliding window
        max_window_size = 60; % Lunghezza della finestra (60 secondi)

        % Sliding window inizializzata
        sliding_window = zeros(max_window_size, 4); % Colonne: speed, rpm, throttle, brake

        simulation_id = randi(1e6); % Genera un numero casuale unico

        % Tempo casuale per iniziare l'attacco
        stop_time = 387;
        attack_started = false;
        attack_time = NaN;

        % Salvataggio della finestra
        saved_window = zeros(1, max_window_size * 2 + 4);

        % Inizializza cost_to_go_list
        if isfile('RL_cost_to_go.mat')
            % Carica il file se esiste
            loaded_data = load('RL_cost_to_go.mat', 'cost_to_go_list');
            cost_to_go_list = loaded_data.cost_to_go_list;
        else
            % Inizializza cost_to_go_list come vuoto
            cost_to_go_list = zeros(0, 2);
            save('RL_cost_to_go.mat', 'cost_to_go_list');
            fprintf('File RL_cost_to_go.mat non trovato. Creato nuovo file.\n');
        end

        initialized = true;
    end

    % Calcolo del **Cost to Go** ad ogni chiamata
    if size(sliding_window, 1) >= max_window_size
        % Recupera stato attuale dalla sliding window
        Speed_1_min_ago = sliding_window(1, 1); % Speed 60 secondi prima
        RPM_1_min_ago = sliding_window(1, 2);  % RPM 60 secondi prima
        throttle_last_60s = sliding_window(:, 3); % Throttle degli ultimi 60 secondi
        brake_last_60s = sliding_window(:, 4);    % Brake degli ultimi 60 secondi
        Speed_now = VehicleSpeed; % Speed attuale
        RPM_now = EngineRPM;     % RPM attuale

        % Crea il vettore di stato
        state_vector = [Speed_1_min_ago, RPM_1_min_ago, ...
                        throttle_last_60s(:)', brake_last_60s(:)', ...
                        Speed_now, RPM_now];

        % Converte il vettore di stato in un array Python
        py_state_vector = py.numpy.array(state_vector);

        % Chiama la funzione Python per calcolare il **Cost to Go**
        cost_to_go = py.RL_MSP_cost_to_go.cost_to_go(py_state_vector);
        cost_to_go = double(cost_to_go); % Converte in double

        % Aggiungi il valore calcolato alla lista e salva nel file
        cost_to_go_list = [cost_to_go_list; time, cost_to_go];
        save('RL_cost_to_go.mat', 'cost_to_go_list'); % Salva nel file
        fprintf('Cost to Go calcolato e salvato: t = %.2f, Cost = %.4f\n', time, cost_to_go);
    end

    % Controlla se le specifiche sono violate
    if VehicleSpeed > 120 || EngineRPM > 4500
        fprintf('Specifiche violate a t = %d secondi.\n', int32(round(time)));
        fprintf('Tempo di violazione: %d secondi.\n', int32(round(time - attack_time)));

        % Calcola il costo come differenza tra tempo di violazione e tempo di attacco
        cost = time - attack_time;

        % Salva la finestra di stato e il costo
        if any(saved_window) && cost < 200
            saved_data = [saved_window, cost]; % Combina la finestra con il costo
        end

        % Imposta terminate a true per segnalare la fine
        terminate = true;
        return; % Termina la funzione
    end

    % Se l'attacco non è iniziato
    if ~attack_started && time >= stop_time
        attack_started = true;
        attack_time = time;
        fprintf('Attacco iniziato a t = %d secondi.\n', int32(round(time)));

        % Salva la finestra di stato
        speed_60s_before = sliding_window(1, 1); % Speed 60 secondi prima
        rpm_60s_before = sliding_window(1, 2);  % RPM 60 secondi prima
        throttle_last_60s = sliding_window(:, 3); % Throttle
        brake_last_60s = sliding_window(:, 4);    % Brake
        saved_window = [speed_60s_before, rpm_60s_before, ...
                        throttle_last_60s(:)', brake_last_60s(:)', ...
                        VehicleSpeed, EngineRPM];
    end

    % Se l'attacco è iniziato
    if attack_started
        if isempty(predicted_pairs) || isempty(pair_index)
            predicted_pairs = zeros(10, 2);
            pair_index = 0;
        end

        if pair_index == 0 || pair_index > 10
            throttle_last_20s = sliding_window(end-19:end, 3);
            brake_last_20s = sliding_window(end-19:end, 4);
            sliding_window_state = [throttle_last_20s, brake_last_20s];
            py_sliding_window = py.numpy.array(sliding_window_state);

            results = py.Attacker.attack(py_sliding_window);
            results = double(results);
            predicted_pairs(:, 1) = max(0, min(100, results(1:10, 1)));
            predicted_pairs(:, 2) = max(0, min(1000, results(1:10, 2)));

            pair_index = 1;
        end

        throttle = round(predicted_pairs(pair_index, 1));
        brake = round(predicted_pairs(pair_index, 2));
        pair_index = pair_index + 1;
    else
        throttle = max(0, mvnrnd(50, 100, 1));
        brake = max(0, mvnrnd(250, 5000, 1));
    end

    % Aggiorna la sliding window
    sliding_window = circshift(sliding_window, -1, 1);
    sliding_window(end, :) = [VehicleSpeed, EngineRPM, throttle, brake];
end
%}


function [throttle, brake] = RL_MSP_controller(EngineRPM, VehicleSpeed, time)
    % Controller chiamato ad ogni step di simulazione (1 secondo)
    % Mantiene i buffer degli ultimi 60 secondi di dati.
    % Le azioni (throttle, brake) restano casuali per tutta la durata.
    % Dopo 60 secondi, chiama Python per calcolare il cost_to_go dello stato attuale
    % (e internamente migliorare la policy), ma non influenza le azioni.

    persistent throttle_buffer brake_buffer speed_buffer rpm_buffer simulation_id initialized 
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
        brake_mean = 250;
        covariance_matrix = [100, -500; -500, 5000];

        % Seed casuale
        simulation_id = randi(1e6); % Genera un numero casuale unico

        % Calcola il valore di posixtime e prendi solo le ultime 4 cifre
        timestamp = round(posixtime(datetime('now')) * 1000);
        seed_suffix = mod(timestamp, 10000); % Prendi le ultime 4 cifre

        % Combina il seed con gli altri parametri
        random_offset = randi(1000); % Offset casuale
        seed = simulation_id + round(time * 1000) + random_offset + seed_suffix;
        
        % Imposta il seed
        rng(seed, 'twister');
        
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
    
    % Calcola il valore di posixtime e prendi solo le ultime 4 cifre
    timestamp = round(posixtime(datetime('now')) * 1000);
    seed_suffix = mod(timestamp, 10000); % Prendi le ultime 4 cifre

    % Combina il seed con gli altri parametri
    random_offset = randi(1000); % Offset casuale
    seed = simulation_id + round(time * 1000) + random_offset + seed_suffix;
    
    % Imposta il seed
    rng(seed, 'twister');
    
    % Generazione di campioni casuali
    inputs = mvnrnd([throttle_mean, brake_mean], covariance_matrix, 1);
    
    throttle = max(0, inputs(1));
    brake = max(0, inputs(2));
    
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