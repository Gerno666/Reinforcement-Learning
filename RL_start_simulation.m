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
figure;

subplot(4, 1, 1);
plot(time_signal, speed_signal, '-b');
title('Velocità (Speed) nel tempo');
xlabel('Tempo (s)');
ylabel('Velocità (km/h)');

subplot(4, 1, 2);
plot(time_signal, rpm_signal, '-r');
title('Giri del motore (RPM) nel tempo');
xlabel('Tempo (s)');
ylabel('RPM');

subplot(4, 1, 3);
plot(time_signal, throttle_signal, '-g');
title('Throttle nel tempo');
xlabel('Tempo (s)');
ylabel('Throttle (%)');

subplot(4, 1, 4);
plot(time_signal, brake_signal, '-k');
title('Brake nel tempo');
xlabel('Tempo (s)');
ylabel('Brake Torque');

% Salva il grafico
saveas(gcf, 'simulation_results_prova_2.png');
fprintf('Risultati della simulazione salvati in simulation_results.png\n');