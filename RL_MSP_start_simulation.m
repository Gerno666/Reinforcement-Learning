% Nome del modello
model_name = 'RL_MSP_FINAL_sldemo_autotrans';

% Nome del file da eliminare
file_to_delete = 'RL_cost_to_go.mat';

% Verifica se il file esiste
if isfile(file_to_delete)
    % Elimina il file
    delete(file_to_delete);
    fprintf('Il file %s è stato eliminato con successo.\n', file_to_delete);
else
    fprintf('Il file %s non esiste.\n', file_to_delete);
end

% Ricrea il file
cost_to_go_list = zeros(0, 2); % Inizializza come vuoto
save(file_to_delete, 'cost_to_go_list');
fprintf('Il file %s è stato ricreato con successo.\n', file_to_delete);
% Durata della simulazione
simulation_time = 1000; % Durata di un'ora (600 secondi)

% Verifica se il modello è aperto
if ~bdIsLoaded(model_name)
    load_system(model_name); % Carica il modello
end

% Configura la simulazione
simIn = Simulink.SimulationInput(model_name);
simIn = simIn.setModelParameter('StopTime', num2str(simulation_time)); % Imposta il tempo di stop
set_param(model_name, 'SignalLogging', 'on'); % Abilita il logging dei segnali

% Avvia la simulazione
fprintf('Inizio simulazione del modello %s per %d secondi...\n', model_name, simulation_time);
simOut = sim(simIn);
fprintf('Simulazione completata.\n');

% Estrai i segnali dai risultati della simulazione
logsout = simOut.get('sldemo_autotrans_output');

% Estrai i segnali necessari
time_signal = logsout.getElement('time').Values.Data; % Tempo
throttle_signal = logsout.getElement('Throttle').Values.Data; % Throttle
brake_signal = logsout.getElement('Brake').Values.Data; % Brake
speed_signal = logsout.getElement('VehicleSpeed').Values.Data; % Velocità
rpm_signal = logsout.getElement('EngineRPM').Values.Data; % RPM

% Carica i valori di Cost to Go dal file RL_cost_to_go.mat
cost_to_go_file = 'RL_cost_to_go.mat';
if isfile(cost_to_go_file)
    loaded_data = load(cost_to_go_file, 'cost_to_go_list');
    cost_to_go_list = loaded_data.cost_to_go_list;

    % Estrai tempo e valori di Cost to Go
    cost_to_go_time_adjusted = cost_to_go_list(:, 1); % Tempo (colonna 1)
    cost_to_go_values_adjusted = cost_to_go_list(:, 2); % Valori Cost to Go (colonna 2)
else
    error('Il file RL_cost_to_go.mat non esiste o non contiene cost_to_go_list.');
end

% Assicura che tutti i segnali abbiano la stessa lunghezza
min_length = min([length(time_signal), length(throttle_signal), ...
                  length(brake_signal), length(speed_signal), ...
                  length(rpm_signal)]);

time_signal = time_signal(1:min_length);
throttle_signal = throttle_signal(1:min_length);
brake_signal = brake_signal(1:min_length);
speed_signal = speed_signal(1:min_length);
rpm_signal = rpm_signal(1:min_length);

% Imposta la finestra del grafico (80% x 80% dello schermo)
screen_size = get(0, 'ScreenSize'); % Ottieni le dimensioni dello schermo
figure_width = screen_size(3) * 0.8; % 80% della larghezza
figure_height = screen_size(4) * 0.8; % 80% dell'altezza
figure('Position', [100, 100, figure_width, figure_height]); % Finestra centrata e ridimensionata

% Plot dei risultati
subplot(5, 1, 1);
plot(time_signal, speed_signal, '-b');
title('Velocità (Speed) nel tempo');
xlabel('Tempo (s)');
ylabel('Velocità (km/h)');

subplot(5, 1, 2);
plot(time_signal, rpm_signal, '-r');
title('Giri del motore (RPM) nel tempo');
xlabel('Tempo (s)');
ylabel('RPM');

subplot(5, 1, 3);
plot(time_signal, throttle_signal, '-g');
title('Throttle nel tempo');
xlabel('Tempo (s)');
ylabel('Throttle (%)');

subplot(5, 1, 4);
plot(time_signal, brake_signal, '-k');
title('Brake nel tempo');
xlabel('Tempo (s)');
ylabel('Brake Torque');

% Trova eventuali discontinuità nei dati temporali
time_diff = diff(cost_to_go_time_adjusted);
threshold = 1.5 * median(time_diff); % Soglia per definire una discontinuità
discontinuity_indices = find(time_diff > threshold);

% Inserisci NaN nei punti di discontinuità per interrompere la linea
cost_to_go_time_fixed = cost_to_go_time_adjusted;
cost_to_go_values_fixed = cost_to_go_values_adjusted;
for idx = flip(discontinuity_indices)' % Inserisci da destra verso sinistra
    cost_to_go_time_fixed = [cost_to_go_time_fixed(1:idx); NaN; cost_to_go_time_fixed(idx+1:end)];
    cost_to_go_values_fixed = [cost_to_go_values_fixed(1:idx); NaN; cost_to_go_values_fixed(idx+1:end)];
end

% Plot senza la linea continua nei punti discontinui
subplot(5, 1, 5);
plot(cost_to_go_time_fixed, cost_to_go_values_fixed, '-m');
title('Cost to Go nel tempo');
xlabel('Tempo (s)');
ylabel('Cost to Go');
xlim([0, simulation_time]); % Imposta il limite delle x per coprire l'intervallo 0-600
grid on; % Aggiungi una griglia per migliorare la leggibilità

% Salva il grafico
saveas(gcf, 'RL_MSP_results_new_R_100_60.png');
fprintf('Risultati della simulazione salvati in RL_MSP_results.png\n');

% Chiudi il modello Simulink
close_system(model_name, 0);