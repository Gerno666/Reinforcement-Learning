%{
% Nome del modello
model_name = 'RL_MSP_FINAL_sldemo_autotrans';

% Nome del file da eliminare
file_to_delete = 'RL_cost_to_go.mat';

% Variabili globali per il monitoraggio
global cpu_usage ram_usage timestamps;
cpu_usage = []; % Percentuale di CPU
ram_usage = []; % RAM usata (MB)
timestamps = []; % Tempo trascorso
tic; % Inizia il timer

% Funzione per monitorare CPU e RAM
function [cpu, ram] = monitor_resources()
    % Ottieni l'utilizzo della CPU
    [~, cpuinfo] = system('ps -A -o %cpu | awk ''{s+=$1} END {print s}'''); % macOS/Linux
    cpu = str2double(cpuinfo);

    % Ottieni l'utilizzo della RAM
    [~, raminfo] = system('vm_stat | grep "Pages active"'); % macOS
    ram_active_pages = sscanf(raminfo, 'Pages active: %d');
    ram = ram_active_pages * 4096 / 1e6; % Converti in MB (pagina di 4 KB)
end

% Funzione per registrare risorse
function log_resources()
    global cpu_usage ram_usage timestamps; % Accedi alle variabili globali
    [cpu, ram] = monitor_resources(); % Ottieni CPU e RAM
    cpu_usage(end+1) = cpu; %#ok<*AGROW>
    ram_usage(end+1) = ram;
    timestamps(end+1) = toc; % Registra il tempo trascorso
end

% Timer per il monitoraggio
monitor_timer = timer('ExecutionMode', 'fixedRate', 'Period', 1, ...
    'TimerFcn', @(~,~) log_resources());

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
simulation_time = 1000; % Durata di un'ora (1000 secondi)

% Verifica se il modello è aperto
if ~bdIsLoaded(model_name)
    load_system(model_name); % Carica il modello
end

% Configura la simulazione
simIn = Simulink.SimulationInput(model_name);
simIn = simIn.setModelParameter('StopTime', num2str(simulation_time)); % Imposta il tempo di stop
set_param(model_name, 'SignalLogging', 'on'); % Abilita il logging dei segnali

% Avvia la simulazione e il monitoraggio
fprintf('Inizio simulazione del modello %s per %d secondi...\n', model_name, simulation_time);
start(monitor_timer); % Avvia il timer di monitoraggio
simOut = sim(simIn); % Esegui la simulazione
stop(monitor_timer); % Ferma il timer
delete(monitor_timer); % Elimina il timer

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

% Genera il grafico delle risorse
cpu_normalized = cpu_usage / feature('numcores'); % Normalizza per il numero di core
figure;
subplot(2, 1, 1);
plot(timestamps, cpu_normalized, '-b', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('CPU Usage per Core (%)');
title('CPU Usage per Core During Simulation');

subplot(2, 1, 2);
plot(timestamps, ram_usage, '-g', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('RAM Usage (MB)');
title('RAM Usage During Simulation');

% Salva il grafico in formato PNG
saveas(gcf, 'CPU_RAM_Monitoring_RL_MSP_FINAL.png');
fprintf('Grafico salvato come CPU_RAM_Monitoring_RL_MSP_FINAL.png\n');

% Chiudi il modello Simulink
close_system(model_name, 0);

fprintf('Monitoraggio completato.\n');

%}

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

% Supponiamo che cost_to_go_time_adjusted sia il tuo array
% Filtra i valori non nulli e maggiori di 0
valid_values = cost_to_go_values_adjusted(cost_to_go_time_adjusted > 0);

% Calcola la media dei valori validi
mean_value = mean(valid_values);

% Calcola la varianza dei valori validi
variance_value = var(valid_values);

% Mostra i risultati
fprintf('La media dei valori non nulli e maggiori di 0 è: %.4f\n', mean_value);
fprintf('La varianza dei valori non nulli e maggiori di 0 è: %.4f\n', variance_value);

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

% Plot of results
subplot(5, 1, 1);
plot(time_signal, speed_signal, '-b');
title('Speed Over Time');
xlabel('Time (s)');
ylabel('Speed (km/h)');

subplot(5, 1, 2);
plot(time_signal, rpm_signal, '-r');
title('Engine RPM Over Time');
xlabel('Time (s)');
ylabel('RPM');

subplot(5, 1, 3);
plot(time_signal, throttle_signal, '-g');
title('Throttle Over Time');
xlabel('Time (s)');
ylabel('Throttle (%)');

subplot(5, 1, 4);
plot(time_signal, brake_signal, '-k');
title('Brake Torque Over Time');
xlabel('Time (s)');
ylabel('Brake Torque (Nm)');

% Find discontinuities in the temporal data
time_diff = diff(cost_to_go_time_adjusted);
threshold = 1.5 * median(time_diff); % Threshold to define a discontinuity
discontinuity_indices = find(time_diff > threshold);

% Insert NaN at discontinuity points to break the line
cost_to_go_time_fixed = cost_to_go_time_adjusted;
cost_to_go_values_fixed = cost_to_go_values_adjusted;
for idx = flip(discontinuity_indices)' % Insert from right to left
    cost_to_go_time_fixed = [cost_to_go_time_fixed(1:idx); NaN; cost_to_go_time_fixed(idx+1:end)];
    cost_to_go_values_fixed = [cost_to_go_values_fixed(1:idx); NaN; cost_to_go_values_fixed(idx+1:end)];
end

% Plot without continuous line at discontinuity points
subplot(5, 1, 5);
plot(cost_to_go_time_fixed, cost_to_go_values_fixed, '-m');
title('Cost to Go Over Time');
xlabel('Time (s)');
ylabel('Cost to Go');
xlim([0, simulation_time]); % Set x-axis limit to cover the range 0-600
grid on; % Add a grid for better readability

% Salva il grafico
saveas(gcf, 'RL_MSP_results_ultimo_1_en.png');
fprintf('Risultati della simulazione salvati in RL_MSP_results.png\n');

% Chiudi il modello Simulink
close_system(model_name, 0);