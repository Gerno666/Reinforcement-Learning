%{
% Nome del modello
model_name = 'RL_sldemo_autotrans';

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

    num_cores = feature('numcores'); % Numero di core disponibili
    
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

% Verifica se il modello è aperto
if ~bdIsLoaded(model_name)
    load_system(model_name); % Carica il modello
end

% Imposta il tempo di simulazione
simulation_time = 1000; % Durata massima della simulazione (in secondi)

% Configura la simulazione
simIn = Simulink.SimulationInput(model_name);
simIn = simIn.setModelParameter('StopTime', num2str(simulation_time)); % Imposta il tempo di stop

% Abilita il logging dei segnali
set_param(model_name, 'SignalLogging', 'on');

% Timer per il monitoraggio
monitor_timer = timer('ExecutionMode', 'fixedRate', 'Period', 1, ...
    'TimerFcn', @(~,~) log_resources());

% Avvia la simulazione e il monitoraggio
fprintf('Inizio simulazione del modello %s...\n', model_name);
start(monitor_timer); % Avvia il timer
simOut = sim(simIn); % Esegui la simulazione
stop(monitor_timer); % Ferma il timer

% Concludi
fprintf('Simulazione completata.\n');

% Elimina il timer
delete(monitor_timer);

% Estrai i segnali dai risultati della simulazione
logsout = simOut.get('sldemo_autotrans_output');
time_signal = logsout.getElement('time').Values.Data; % Tempo
throttle_signal = logsout.getElement('Throttle').Values.Data; % Throttle
brake_signal = logsout.getElement('Brake').Values.Data; % Brake
speed_signal = logsout.getElement('VehicleSpeed').Values.Data; % Velocità
rpm_signal = logsout.getElement('EngineRPM').Values.Data; % RPM

% Assicura che i segnali abbiano la stessa lunghezza
min_length = min([length(time_signal), length(throttle_signal), ...
                  length(brake_signal), length(speed_signal), length(rpm_signal)]);

time_signal = time_signal(1:min_length);
throttle_signal = throttle_signal(1:min_length);
brake_signal = brake_signal(1:min_length);
speed_signal = speed_signal(1:min_length);
rpm_signal = rpm_signal(1:min_length);

cpu_normalized = cpu_usage / 8;

% Genera il grafico delle risorse
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
saveas(gcf, 'CPU_RAM_attacker.png');
fprintf('Grafico salvato come resource_usage.png\n');

fprintf('Monitoraggio completato.\n');
%}

%{
% Nome del modello
model_name = 'RL_sldemo_autotrans';

% Verifica se il modello è aperto
if ~bdIsLoaded(model_name)
    load_system(model_name); % Carica il modello
end

% Imposta il tempo di simulazione
simulation_time = 1000; % Durata massima della simulazione (in secondi)

% Configura la simulazione
simIn = Simulink.SimulationInput(model_name);
simIn = simIn.setModelParameter('StopTime', num2str(simulation_time)); % Imposta il tempo di stop

% Abilita il logging dei segnali
set_param(model_name, 'SignalLogging', 'on');

% Avvia la simulazione
fprintf('Inizio simulazione del modello %s...\n', model_name);
simOut = sim(simIn);

% Concludi
fprintf('Simulazione completata.\n');

% Estrai i segnali dai risultati della simulazione
logsout = simOut.get('sldemo_autotrans_output');
time_signal = logsout.getElement('time').Values.Data; % Tempo
throttle_signal = logsout.getElement('Throttle').Values.Data; % Throttle
brake_signal = logsout.getElement('Brake').Values.Data; % Brake
speed_signal = logsout.getElement('VehicleSpeed').Values.Data; % Velocità
rpm_signal = logsout.getElement('EngineRPM').Values.Data; % RPM

% Assicura che i segnali abbiano la stessa lunghezza
min_length = min([length(time_signal), length(throttle_signal), ...
                  length(brake_signal), length(speed_signal), length(rpm_signal)]);

time_signal = time_signal(1:min_length);
throttle_signal = throttle_signal(1:min_length);
brake_signal = brake_signal(1:min_length);
speed_signal = speed_signal(1:min_length);
rpm_signal = rpm_signal(1:min_length);

% Funzione per aggiustare i segnali in chunks
function adjusted_signal = adjust_sequence_in_chunks(signal, target_mean, target_variance, chunk_size)
    num_chunks = floor(length(signal) / chunk_size);
    adjusted_signal = zeros(size(signal));

    for i = 1:num_chunks
        start_idx = (i-1) * chunk_size + 1;
        end_idx = start_idx + chunk_size - 1;
        chunk = signal(start_idx:end_idx);

        % Calcolo della media e varianza corrente
        current_mean = mean(chunk);
        current_variance = var(chunk);

        % Aggiustamento del chunk
        chunk = (chunk - current_mean) / sqrt(current_variance); % Normalizzazione
        chunk = chunk * sqrt(target_variance) + target_mean; % Scalatura e traslazione

        % Clipping ai limiti fisici
        if target_mean == 50
            chunk = min(max(chunk, 0), 100); % Limiti per Throttle
        else
            chunk = min(max(chunk, 0), 1000); % Limiti per Brake
        end

        adjusted_signal(start_idx:end_idx) = chunk;
    end

    % Gestione del chunk rimanente
    remaining = mod(length(signal), chunk_size);
    if remaining > 0
        start_idx = num_chunks * chunk_size + 1;
        chunk = signal(start_idx:end);

        % Calcolo della media e varianza corrente
        current_mean = mean(chunk);
        current_variance = var(chunk);

        % Aggiustamento del chunk
        chunk = (chunk - current_mean) / sqrt(current_variance); % Normalizzazione
        chunk = chunk * sqrt(target_variance) + target_mean; % Scalatura e traslazione

        % Clipping ai limiti fisici
        if target_mean == 50
            chunk = min(max(chunk, 0), 100); % Limiti per Throttle
        else
            chunk = min(max(chunk, 0), 1000); % Limiti per Brake
        end

        adjusted_signal(start_idx:end) = chunk;
    end
end

% Applica variazioni di RPM e aggiustamenti su Throttle e Brake dopo il tempo 320
idx_320 = find(time_signal >= 320, 1);

if ~isempty(idx_320)
    % Variazioni RPM nel range ±10
    rpm_signal(idx_320:end) = rpm_signal(idx_320:end) + randi([-100, 100], size(rpm_signal(idx_320:end)));

    speed_signal(idx_320:end-4) = speed_signal(idx_320:end-4) + randi([-1, 1], size(speed_signal(idx_320:end-4)));

    % Aggiusta Throttle e Brake
    throttle_signal(idx_320:end-1) = adjust_sequence_in_chunks(throttle_signal(idx_320:end-1), 50, 100, 10);
    brake_signal(idx_320:end-1) = adjust_sequence_in_chunks(brake_signal(idx_320:end-1), 250, 5000, 10);
    throttle_signal(end) = NaN;
    brake_signal(end) = NaN;
end

% Genera il grafico
figure('Units', 'normalized', 'Position', [0.2, 0.1, 0.6, 0.8]);

% Speed plot with violation line
subplot(4, 1, 1);
plot(time_signal, speed_signal, '-b');
hold on;
yline(120, '--r', 'Speed Limit (120 km/h)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left'); % Speed limit line
hold off;
title('Speed over Time');
xlabel('Time (s)');
ylabel('Speed (km/h)');

% RPM plot with violation line
subplot(4, 1, 2);
plot(time_signal, rpm_signal, '-b');
hold on;
yline(4500, '--r', 'RPM Limit (4500)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left'); % RPM limit line
hold off;
title('Engine RPM over Time');
xlabel('Time (s)');
ylabel('RPM');

% Throttle plot
subplot(4, 1, 3);
plot(time_signal, throttle_signal, '-g'); % Linea verde più chiara e spessa
title('Throttle over Time');
xlabel('Time (s)');
ylabel('Throttle (%)');

% Brake plot
subplot(4, 1, 4);
plot(time_signal, brake_signal, '-k'); % Linea nera più spessa
title('Brake Torque over Time');
xlabel('Time (s)');
ylabel('Brake Torque (Nm)');

% Salva il grafico
% saveas(gcf, 'simulation_results_adjusted.png');
fprintf('Risultati della simulazione salvati in simulation_results_adjusted.png\n');
%}



% Nome del modello
model_name = 'RL_sldemo_autotrans';

% Verifica se il modello è aperto
if ~bdIsLoaded(model_name)
    load_system(model_name); % Carica il modello
end

% Imposta il tempo di simulazione
simulation_time = 1000; % Durata massima della simulazione (in secondi)

% Configura la simulazione
simIn = Simulink.SimulationInput(model_name);
simIn = simIn.setModelParameter('StopTime', num2str(simulation_time)); % Imposta il tempo di stop

% Abilita il logging dei segnali
set_param(model_name, 'SignalLogging', 'on');

% Avvia la simulazione
fprintf('Inizio simulazione del modello %s...\n', model_name);
simOut = sim(simIn);

% Concludi
fprintf('Simulazione completata.\n');

% Estrai i segnali dai risultati della simulazione
logsout = simOut.get('sldemo_autotrans_output');
time_signal = logsout.getElement('time').Values.Data; % Tempo
throttle_signal = logsout.getElement('Throttle').Values.Data; % Throttle
brake_signal = logsout.getElement('Brake').Values.Data; % Brake
speed_signal = logsout.getElement('VehicleSpeed').Values.Data; % Velocità
rpm_signal = logsout.getElement('EngineRPM').Values.Data; % RPM

% Assicura che i segnali abbiano la stessa lunghezza
min_length = min([length(time_signal), length(throttle_signal), ...
                  length(brake_signal), length(speed_signal), length(rpm_signal)]);

time_signal = time_signal(1:min_length);
throttle_signal = throttle_signal(1:min_length);
brake_signal = brake_signal(1:min_length);
speed_signal = speed_signal(1:min_length);
rpm_signal = rpm_signal(1:min_length);

% Genera il grafico
figure('Units', 'normalized', 'Position', [0.2, 0.1, 0.6, 0.8]);

% Speed plot with violation line
subplot(4, 1, 1);
plot(time_signal, speed_signal, '-b');
hold on;
yline(120, '--r', 'Speed Limit (120 km/h)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left'); % Speed limit line
hold off;
title('Speed over Time');
xlabel('Time (s)');
ylabel('Speed (km/h)');

% RPM plot with violation line
subplot(4, 1, 2);
plot(time_signal, rpm_signal, '-b');
hold on;
yline(4500, '--r', 'RPM Limit (4500)', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'left'); % RPM limit line
hold off;
title('Engine RPM over Time');
xlabel('Time (s)');
ylabel('RPM');

% Throttle plot
subplot(4, 1, 3);
plot(time_signal, throttle_signal, '-g');
title('Throttle over Time');
xlabel('Time (s)');
ylabel('Throttle (%)');

% Brake plot
subplot(4, 1, 4);
plot(time_signal, brake_signal, '-k');
title('Brake Torque over Time');
xlabel('Time (s)');
ylabel('Brake Torque (Nm)');

% Salva il grafico
saveas(gcf, 'simulation_results_mean.png');
fprintf('Risultati della simulazione salvati in simulation_results.png\n');