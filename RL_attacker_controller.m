function [throttle, brake, terminate] = RL_attacker_controller(EngineRPM, VehicleSpeed, time)
    % Inizializza le variabili di output
    throttle = 0;
    brake = 0;
    terminate = false; % Segnale per terminare la simulazione

    % Parametri globali
    persistent sliding_window stop_time initialized max_window_size attack_started attack_time saved_window
    persistent predicted_pairs pair_index simulation_id

    % Dichiarazione estrinseca della funzione Python
    coder.extrinsic('py.numpy.array');
    coder.extrinsic('py.Attacker.attack');

    % Inizializza variabili alla prima chiamata
    if isempty(initialized)
        % Dimensione della sliding window
        max_window_size = 60; % Lunghezza della finestra (60 secondi)

        % Sliding window inizializzata
        sliding_window = zeros(max_window_size, 4); % Colonne: speed, rpm, throttle, brake

        simulation_id = randi(1e6); % Genera un numero casuale unico

        % Calcola il valore di posixtime e prendi solo le ultime 4 cifre
        timestamp = round(posixtime(datetime('now')) * 1000);
        seed_suffix = mod(timestamp, 10000); % Prendi le ultime 4 cifre

        % Combina il seed con gli altri parametri
        random_offset = randi(1000); % Offset casuale
        seed = simulation_id + round(time * 1000) + random_offset + seed_suffix;
        
        % Imposta il seed
        rng(seed, 'twister');

        % Tempo casuale per iniziare l'attacco
        stop_time = randi([61, 360]);

        % Flag e tempi dell'attacco
        attack_started = false;
        attack_time = NaN;

        % Salvataggio della finestra
        saved_window = zeros(1, max_window_size * 2 + 4);

        initialized = true;
    end


    % Controlla se le specifiche sono violate
    if VehicleSpeed > 120 || EngineRPM > 4500
        fprintf('Specifiche violate a t = %d secondi.\n', int32(round(time)));
        fprintf('Tempo di violazione: %d secondi.\n', int32(round(time - attack_time)));

        % Calcola il costo come differenza tra tempo di violazione e tempo di attacco
        cost = time - attack_time;

        % Salva la finestra di stato e il costo chiamando la funzione esterna
        if any(saved_window) && cost < 200
            saved_data = [saved_window, cost]; % Combina la finestra con il costo
            %update_simulation_data(saved_data); % Chiama la funzione per salvare i dati
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
        speed_60s_before = sliding_window(1, 1); % Speed 60 secondi prima (primo elemento della finestra)
        rpm_60s_before = sliding_window(1, 2);  % RPM 60 secondi prima (primo elemento della finestra)
        throttle_last_60s = sliding_window(:, 3); % Throttle degli ultimi 60 secondi
        brake_last_60s = sliding_window(:, 4);    % Brake degli ultimi 60 secondi
    
        saved_window = [speed_60s_before, rpm_60s_before, ... % Speed e RPM di 60 secondi prima
                        throttle_last_60s(:)', ...           % Throttle degli ultimi 60 secondi
                        brake_last_60s(:)', ...             % Brake degli ultimi 60 secondi
                        round(VehicleSpeed), round(EngineRPM)]; % Speed e RPM attuali
    end

    % Se l'attacco è iniziato
    if attack_started
    
        % Inizializza le variabili alla prima iterazione dell'attacco
        if isempty(predicted_pairs) || isempty(pair_index)
            predicted_pairs = zeros(600, 2); % Per contenere le 10 coppie predette
            pair_index = 0;  % Indice iniziale
        end
    
        % Se tutte le coppie sono state consumate, chiama la funzione Python
        if pair_index == 0 || pair_index > 10
            % Salva la finestra di stato (ultimi 20 secondi)
            throttle_last_20s = sliding_window(end-19:end, 3); % Throttle degli ultimi 20 secondi
            brake_last_20s = sliding_window(end-19:end, 4);    % Brake degli ultimi 20 secondi
            
            % Combina throttle e brake in un unico array con dimensione [20 x 2]
            sliding_window_state = [throttle_last_20s, brake_last_20s];
            
            % Converte in array NumPy
            py_sliding_window = py.numpy.array(sliding_window_state);
            
            % Chiama la funzione Python per ottenere throttle e brake
            results = py.Attacker.attack(py_sliding_window);
    
            % Verifica i risultati della funzione Python
            if isempty(results) || any(isnan(double(results)))
                fprintf('Errore nella funzione Python: throttle e brake impostati a 0.\n');
                predicted_pairs = zeros(10, 2); % Imposta tutte le coppie a zero
            else
                results = double(results); % Converte l'output in array numerico MATLAB
                % Estrai e assegna i valori di throttle e brake
                throttle_values = results(1:10, 1);  % Colonna 1: Throttle
                brake_values = results(1:10, 2);     % Colonna 2: Brake
                
                % Clampa i valori ai rispettivi range
                predicted_pairs(:, 1) = max(0, min(100, throttle_values)); % Throttle tra 0 e 100
                predicted_pairs(:, 2) = max(0, min(1000, brake_values));   % Brake tra 0 e 1000
            end
    
            % Reimposta il contatore
            pair_index = 1;
        end

        % Assegna la coppia corrente di throttle e brake
        throttle = round(predicted_pairs(pair_index, 1));
        brake = round(predicted_pairs(pair_index, 2));
    
        % Incrementa il contatore per la prossima iterazione
        pair_index = pair_index + 1;

        disp(throttle);
        disp(brake);
    else

        % Calcola il valore di posixtime e prendi solo le ultime 4 cifre
        timestamp = round(posixtime(datetime('now')) * 1000);
        seed_suffix = mod(timestamp, 10000); % Prendi le ultime 4 cifre

        % Combina il seed con gli altri parametri
        random_offset = randi(1000); % Offset casuale
        seed = simulation_id + round(time * 1000) + random_offset + seed_suffix;
        
        % Imposta il seed
        rng(seed, 'twister');

        % Definizione della media e della matrice di covarianza
        throttle_mean = 50;
        brake_mean = 250;
        
        % Valore della covarianza negativa
        covariance_matrix = [100, -500; -500, 5000];
        
        % Generazione di campioni casuali
        inputs = mvnrnd([throttle_mean, brake_mean], covariance_matrix, 1);
        
        throttle = max(0, inputs(1));
        brake = max(0, inputs(2));
     
    end

    % Aggiorna la sliding window
    sliding_window = circshift(sliding_window, -1, 1);
    sliding_window(end, :) = [round(VehicleSpeed), round(EngineRPM), throttle, brake];
end


function update_simulation_data(saved_data)
    % Nome del file MAT
    filename = 'RL_10k_simulations_data.mat';

    % Verifica se il file esiste
    if isfile(filename)
        % Carica i dati esistenti
        data = load(filename);
        if isfield(data, 'all_data')
            all_data = data.all_data;
        else
            error('Variabile "all_data" non presente nel file MAT.');
        end
    else
        error('File MAT non trovato. Assicurarsi che il file sia inizializzato correttamente.');
    end

    % Trova la prima riga vuota (NaN) in all_data
    empty_row_index = find(isnan(all_data(:, 1)), 1);

    if isempty(empty_row_index)
        error('Matrice "all_data" piena. Aumentare la dimensione durante l inizializzazione.');
    end

    % Inserisci i nuovi dati nella prima riga vuota
    all_data(empty_row_index, :) = saved_data;

    % Salva i dati aggiornati nel file MAT
    save(filename, 'all_data');
    disp('Dati aggiornati nel file RL_simulation_data.mat');
end