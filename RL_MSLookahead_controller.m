function [throttle, brake, terminate] = RL_MSLookahead_controller(EngineRPM, VehicleSpeed, time)

    % Persistent variables to maintain state across function calls
    persistent throttle_buffer brake_buffer speed_buffer rpm_buffer initialized
    persistent throttle_mean brake_mean covariance_matrix
    persistent terminate_flag
    persistent L start_time states costs steps
    persistent step_history_array

    % Import extrinsic Python functions for policy evaluation and training
    coder.extrinsic('py.RL_MSLookahead_policy.evaluate_cost');
    coder.extrinsic('py.RL_MSLookahead_policy.train_policy');

    % Initialize variables
    if isempty(initialized)
        
        throttle_buffer = zeros(60, 1);
        brake_buffer = zeros(60, 1);
        speed_buffer = zeros(60, 1);
        rpm_buffer = zeros(60, 1);

        throttle_mean = 50;
        brake_mean = 300;
        covariance_matrix = [100, -600; -600, 8000];

        terminate_flag = false;
        L = 10; % Lookahead steps
        start_time = NaN;

        % Initialize persistent data structures
        states = zeros(0, 125); % Assume 125 features for state vector
        costs = zeros(0, 1); % Array vuoto inizializzato come double
        steps = zeros(0, 1); % Array vuoto inizializzato come double

        % Inizializza un array di dimensione 100 per tenere traccia degli step
        step_history_array = nan(100, 1); 

        initialized = true;

    end

    % **Aggiorna i buffer con gli ultimi dati**
    speed_buffer = [speed_buffer; VehicleSpeed];
    rpm_buffer = [rpm_buffer; EngineRPM];
    if length(speed_buffer) > 60
        speed_buffer(1) = [];
        rpm_buffer(1) = [];
    end

    % **Generazione di Throttle e Brake casuali**
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

    % Update throttle and brake buffers
    throttle_buffer = [throttle_buffer; throttle];
    brake_buffer = [brake_buffer; brake];
    if length(throttle_buffer) > 60
        throttle_buffer(1) = [];
        brake_buffer(1) = [];
    end


    % **Multistep Lookahead Logic**
    if (mod(time, 30) == 0 && time >= 61 && isnan(start_time)) || (~isnan(start_time) && time < start_time + L + 1)
        if isnan(start_time)
            start_time = time; % Start new lookahead
            fprintf('Inizio Multistep Lookahead al tempo %d secondi...\n', int32(round(time)));
        end

        Speed_1_min_ago = speed_buffer(1);
        RPM_1_min_ago = rpm_buffer(1);
        last_60_throttle = throttle_buffer(:)';
        last_60_brake = brake_buffer(:)';
        Speed_now = speed_buffer(end);
        RPM_now = rpm_buffer(end);

        state_vector = [Speed_1_min_ago, RPM_1_min_ago, ...
                        last_60_throttle, last_60_brake, ...
                        Speed_now, RPM_now];

        % Evaluate cost for the current state
        current_cost = 0; % Preinizializza current_cost come double
        current_cost = py.RL_MSLookahead_policy.evaluate_cost(state_vector);
        current_cost = double(current_cost);
        adjusted_cost = current_cost + (time - start_time); % Penalità temporale

        % Append the current state, cost, and step
        states = [states; state_vector];
        costs = [costs; adjusted_cost];
        steps = [steps; time - start_time];

        % If the lookahead is complete
        if time == start_time + L
            fprintf('Fine Multistep Lookahead al tempo %d secondi. Analisi dei risultati...\n', int32(round(time)));
        
            % Find the minimum cost and corresponding step
            [~, idx_min] = min(costs);
            best_cost = costs(idx_min);
            best_step = steps(idx_min);
        
            % Verifica se il best_step è maggiore di 0
            if best_step > 0
                fprintf('Aggiornamento policy con costo %.2f dallo step %d...\n', best_cost, int32(best_step));
                
                % Update the policy using the initial state and best cost
                initial_state = states(1, :); % Lo stato iniziale è il primo elemento di `states`
                py.RL_MSLookahead_policy.train_policy(initial_state, best_cost);
            else
                fprintf('Nessun aggiornamento della policy, step %d.\n', int32(best_step));
            end
        
            % Aggiorna lo storico degli step
            step_history_array = [step_history_array(2:end); best_step];
        
            % Conta gli ultimi zeri consecutivi
            consecutive_zeros = 0;
            for i = length(step_history_array):-1:1
                if step_history_array(i) == 0
                    consecutive_zeros = consecutive_zeros + 1;
                else
                    break; % Interrompi il ciclo se trovi un valore diverso da zero
                end
            end
        
            % Stampa il numero di step 0 consecutivi
            fprintf('Step 0 scelto consecutivamente %d volte.\n', int32(consecutive_zeros));
        
            % Verifica la condizione di terminazione
            if consecutive_zeros >= 100
                terminate_flag = true;
                fprintf('Simulazione terminata: step 0 scelto per 50 iterazioni consecutive.\n');
            end
        
            % Reset persistent data
            states = zeros(0, 125);
            costs = zeros(0, 1); % Array vuoto inizializzato come double;
            steps = zeros(0, 1); % Array vuoto inizializzato come double;
            start_time = NaN;
        end
    end

    % Return the termination flag
    terminate = terminate_flag;
end