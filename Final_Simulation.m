% Nome del modello
model_name = 'RL_MSP_FINAL_sldemo_autotrans';

% Nome del file da eliminare
file_to_delete = 'RL_cost_to_go.mat';

% Verifica se il file esiste
if isfile(file_to_delete)
    delete(file_to_delete);
    fprintf('The file %s has been successfully deleted.\n', file_to_delete);
else
    fprintf('The file %s does not exist.\n', file_to_delete);
end

% Ricrea il file
cost_to_go_list = zeros(0, 2); % Inizializza come vuoto
save(file_to_delete, 'cost_to_go_list');
fprintf('The file %s has been successfully recreated.\n', file_to_delete);

% Durata della simulazione
simulation_time = 500;

% Verifica se il modello è aperto
if ~bdIsLoaded(model_name)
    load_system(model_name); % Carica il modello
end

% Configura la simulazione
simIn = Simulink.SimulationInput(model_name);
simIn = simIn.setModelParameter('StopTime', num2str(simulation_time)); % Imposta il tempo di stop
set_param(model_name, 'SignalLogging', 'on'); % Abilita il logging dei segnali

% Avvia la simulazione
fprintf('Starting the simulation of the model %s for %d seconds...\n', model_name, simulation_time);
simOut = sim(simIn);
fprintf('Simulation completed.\n');

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
    cost_to_go_time = cost_to_go_list(:, 1); % Tempo (colonna 1)
    cost_to_go_values = cost_to_go_list(:, 2); % Valori Cost to Go (colonna 2)
else
    error('The file RL_cost_to_go.mat does not exist or does not contain cost_to_go_list.');
end

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
idx_387 = find(time_signal >= 387, 1);

if ~isempty(idx_387)
    % Variazioni RPM nel range ±10
    rpm_signal(idx_387:end) = rpm_signal(idx_387:end) + randi([-100, 100], size(rpm_signal(idx_387:end)));

    speed_signal(idx_387:end-4) = speed_signal(idx_387:end-4) + randi([-1, 1], size(speed_signal(idx_387:end-4)));

    % Aggiusta Throttle e Brake
    throttle_signal(idx_387:end-1) = adjust_sequence_in_chunks(throttle_signal(idx_387:end-1), 50, 100, 10);
    brake_signal(idx_387:end-1) = adjust_sequence_in_chunks(brake_signal(idx_387:end-1), 250, 5000, 10);
    throttle_signal(end) = NaN;
    brake_signal(end) = NaN;
end

% Uniformare i segnali fino a 450 secondi
uniform_time = 0:1:450; % Tempo uniforme da 0 a 450
speed_interp = interp1(time_signal, speed_signal, uniform_time, 'linear', NaN);
rpm_interp = interp1(time_signal, rpm_signal, uniform_time, 'linear', NaN);
throttle_interp = interp1(time_signal, throttle_signal, uniform_time, 'linear', NaN);
brake_interp = interp1(time_signal, brake_signal, uniform_time, 'linear', NaN);
cost_to_go_interp = interp1(cost_to_go_time, cost_to_go_values, uniform_time, 'linear', NaN);

% Trova il primo istante in cui il Cost to Go è minore di 40
threshold = 40;
first_threshold_index = find(cost_to_go_interp < threshold, 1);

if ~isempty(first_threshold_index)
    cutoff_time = uniform_time(first_threshold_index); % Tempo in cui Cost to Go è < 33
    fprintf('Cost to Go drops below %d at time %.2f seconds.\n', threshold, cutoff_time);

    % Inserisci NaN per interrompere i segnali dopo cutoff_time
    speed_interp(first_threshold_index+1:end) = NaN;
    rpm_interp(first_threshold_index+1:end) = NaN;
    throttle_interp(first_threshold_index+1:end) = NaN;
    brake_interp(first_threshold_index+1:end) = NaN;
    cost_to_go_interp(first_threshold_index+1:end) = NaN;
else
    cutoff_time = max(uniform_time); % Se non trovato, usa il tempo massimo
    fprintf('Cost to Go never drops below %d during the simulation.\n', threshold);
end

% Imposta la finestra del grafico
screen_size = get(0, 'ScreenSize'); % Ottieni le dimensioni dello schermo
figure_width = screen_size(3) * 0.8; % 80% della larghezza
figure_height = screen_size(4) * 0.8; % 80% dell'altezza
figure('Position', [100, 100, figure_width, figure_height]); % Finestra centrata e ridimensionata

% Plot dei risultati
subplot(5, 1, 1);
plot(uniform_time, speed_interp, '-b');
hold on;
xline(387, '--r', 'Attack Started', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
xlim([0, 450]);
hold off;
title('Vehicle Speed Over Time');
xlabel('Time (s)');
ylabel('Speed (km/h)');

subplot(5, 1, 2);
plot(uniform_time, rpm_interp, '-r');
hold on;
xline(387, '--r', 'Attack Started', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
xlim([0, 450]);
hold off;
title('Engine RPM Over Time');
xlabel('Time (s)');
ylabel('RPM');

subplot(5, 1, 3);
plot(uniform_time, throttle_interp, '-g');
hold on;
xline(387, '--r', 'Attack Started', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
xlim([0, 450]);
hold off;
title('Throttle Over Time');
xlabel('Time (s)');
ylabel('Throttle (%)');

subplot(5, 1, 4);
plot(uniform_time, brake_interp, '-k');
hold on;
xline(387, '--r', 'Attack Started', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
xlim([0, 450]);
hold off;
title('Brake Torque Over Time');
xlabel('Time (s)');
ylabel('Brake Torque (Nm)');

% Plot del Cost to Go con linea limite
subplot(5, 1, 5);
plot(uniform_time, cost_to_go_interp, '-m');
hold on;
yline(threshold, '--r', 'Threshold', 'LineWidth', 1.5, 'LabelHorizontalAlignment', 'right');
xline(387, '--r', 'Attack Started', 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
xlim([0, 450]);
hold off;
title('Cost to Go Over Time');
xlabel('Time (s)');
ylabel('Cost to Go');
grid on;

% Salva il grafico
saveas(gcf, 'RL_MSP_cost_to_go_attack_uniform.png');
fprintf('Simulation results saved in RL_MSP_cost_to_go_attack_uniform.png\n');

% Chiudi il modello Simulink
close_system(model_name, 0);