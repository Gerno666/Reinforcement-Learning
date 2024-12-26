function [throttle, brake] = RL_window_controller(EngineRPM, VehicleSpeed, time)
    % Parametri globali
    persistent sliding_window stop_time initialized max_window_size

    % Inizializza variabili alla prima chiamata
    if isempty(initialized)
        % Dimensione della sliding window
        max_window_size = 60; % Lunghezza della finestra (60 secondi)

        % Sliding window
        sliding_window = struct('time', zeros(max_window_size, 1), ...
                                'speed', zeros(max_window_size, 1), ...
                                'rpm', zeros(max_window_size, 1), ...
                                'throttle', zeros(max_window_size, 1), ...
                                'brake', zeros(max_window_size, 1));

        % Tempo casuale per terminare la simulazione
        stop_time = randi([60, 120]);

        initialized = true;
    end

    % Genera valori casuali interi di throttle e brake
    throttle_mean = 50;
    brake_mean = 500;
    covariance_matrix = [30, -15; -15, 400];
    
    inputs = mvnrnd([throttle_mean, brake_mean], covariance_matrix, 1);
    
    % Imposta throttle e brake garantendo mutua esclusione
    if rand > 0.5
        % throttle > 0, brake = 0
        throttle = max(0, round(inputs(1)));
        brake = 0;
    else
        % brake > 0, throttle = 0
        throttle = 0;
        brake = max(0, round(inputs(2)));
    end

    % Arrotonda i valori di EngineRPM e VehicleSpeed a interi
    VehicleSpeed = round(VehicleSpeed);
    EngineRPM = round(EngineRPM);

    % Aggiorna la sliding window (mantieni solo gli ultimi 60 valori)
    sliding_window.time = circshift(sliding_window.time, -1);
    sliding_window.speed = circshift(sliding_window.speed, -1);
    sliding_window.rpm = circshift(sliding_window.rpm, -1);
    sliding_window.throttle = circshift(sliding_window.throttle, -1);
    sliding_window.brake = circshift(sliding_window.brake, -1);

    sliding_window.time(end) = time;
    sliding_window.speed(end) = VehicleSpeed;
    sliding_window.rpm(end) = EngineRPM;
    sliding_window.throttle(end) = throttle;
    sliding_window.brake(end) = brake;

    % Controlla se il tempo ha raggiunto il limite per terminare la simulazione
    if time >= stop_time
        % Esporta i dati della sliding window nel workspace
        assignin('base', 'sliding_window', sliding_window);

        % Filtra i valori validi della sliding window
        valid_indices = sliding_window.time > 0;

        % Crea i grafici alla fine della simulazione
        figure;
        subplot(4, 1, 1);
        plot(sliding_window.time(valid_indices), sliding_window.speed(valid_indices));
        title('Velocita nella sliding window');
        xlabel('Tempo (s)');
        ylabel('Speed (km/h)');

        subplot(4, 1, 2);
        plot(sliding_window.time(valid_indices), sliding_window.rpm(valid_indices));
        title('RPM nella sliding window');
        xlabel('Tempo (s)');
        ylabel('RPM');

        subplot(4, 1, 3);
        plot(sliding_window.time(valid_indices), sliding_window.throttle(valid_indices));
        title('Throttle nella sliding window');
        xlabel('Tempo (s)');
        ylabel('Throttle (%)');

        subplot(4, 1, 4);
        plot(sliding_window.time(valid_indices), sliding_window.brake(valid_indices));
        title('Brake nella sliding window');
        xlabel('Tempo (s)');
        ylabel('Brake Torque');

        % Salva i grafici in un file PNG
        saveas(gcf, 'RL_sliding_window_results.png');

        % Interrompi la simulazione
        error('Simulazione terminata a t = %d secondi.', time);
    end
end