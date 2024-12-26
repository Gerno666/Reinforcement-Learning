% Nome del modello
model_name = 'RL_sldemo_autotrans';

% Nome del file MAT
filename = 'RL_10k_simulations_data.mat';

% Numero massimo di righe nel file
max_rows = 10000;

% Loop finché ci sono righe vuote nel file
while true
    % Carica il file MAT
    if isfile(filename)
        load(filename, 'all_data');
        if ~exist('all_data', 'var')
            error('Il file MAT esiste ma non contiene la variabile "all_data".');
        end
    else
        error('File MAT non trovato. Assicurati che il file esista e sia inizializzato correttamente.');
    end

    % Conta il numero di righe vuote
    num_empty_rows = sum(all(isnan(all_data), 2));
    fprintf('Numero di righe vuote nel file: %d\n', num_empty_rows);

    % Termina il ciclo se non ci sono righe vuote
    if num_empty_rows == 0
        fprintf('Tutte le righe sono state riempite. Simulazioni completate.\n');
        break;
    end

    % Imposta il tempo massimo di simulazione
    simulation_time = 6000; % Durata massima della simulazione (in secondi)

    % Carica il modello
    load_system(model_name);

    % Configura la simulazione
    simIn = Simulink.SimulationInput(model_name);
    simIn = simIn.setModelParameter('StopTime', num2str(simulation_time));

    % Abilita il logging dei segnali
    set_param(model_name, 'SignalLogging', 'on');

    % Avvia la simulazione
    try
        simOut = sim(simIn);

        % Controlla il flag di terminazione
        terminate_flag = evalin('base', 'exist(''terminate'', ''var'') && terminate');
        if terminate_flag
            fprintf('Simulazione terminata a causa di violazione delle specifiche.\n');
        end

    catch ME
        fprintf('Errore durante la simulazione: %s\n', ME.message);
    end

    % Chiudi il modello per liberare memoria
    close_system(model_name, 0);
end