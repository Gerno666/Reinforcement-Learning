function speed_gain = Attacker_simulate_speed(throttle, brake)
    % Simula il guadagno di velocità basato sulle coppie di throttle e brake.
    % Args:
    %   throttle (double array): Valori di throttle [10 x 1].
    %   brake (double array): Valori di brake [10 x 1].
    % Returns:
    %   speed_gain (double): Guadagno cumulativo di velocità.

    % Verifica le dimensioni degli input
    if length(throttle) ~= 600 || length(brake) ~= 600
        error('Throttle e Brake devono avere 600 elementi ciascuno.');
    end

    % Nome del modello Simulink
    model_name = 'sldemo_autotrans_test';

    % Carica il modello
    load_system(model_name);

    % Configura i parametri di simulazione
    set_param(model_name, 'StopTime', '900'); % Tempo massimo di simulazione (900 secondi)

    % Configura le prime 100 coppie come costanti
    time_initial = linspace(0, 300, 300)'; % Tempo da 0 a 300 secondi
    throttle_initial = 50 * ones(300, 1); % Throttle costante
    brake_initial = 250 * ones(300, 1); % Brake costante

    % Configura le ultime 10 coppie come dinamiche
    time_generated = linspace(301, 900, 600)'; % Tempo da 301 a 900 secondi
    throttle_generated = throttle(:); % Assicurati che throttle sia una colonna
    brake_generated = brake(:); % Assicurati che brake sia una colonna

    % Combina le prime 100 coppie con le ultime 10 coppie
    time = [time_initial; time_generated];
    throttle_data = [throttle_initial; throttle_generated];
    brake_data = [brake_initial; brake_generated];

    % Crea input completo
    external_input = [time, throttle_data, brake_data];

    % Configura la simulazione con gli input
    in = Simulink.SimulationInput(model_name);
    in = in.setExternalInput(mat2str(external_input));

    % Configura i parametri di logging
    set_param(model_name, 'SignalLogging', 'on');
    set_param(model_name, 'SignalLoggingName', 'sldemo_autotrans_output');

    try
        % Esegui la simulazione
        sim_out = sim(in);

        % Estrai i segnali loggati da logsout
        logsout = sim_out.sldemo_autotrans_output;
        speed_signal = logsout.getElement('VehicleSpeed').Values.Data;

        % Calcola il guadagno cumulativo di velocità
        speed_gain = speed_signal(end);

    catch ME
        % Gestisci gli errori della simulazione
        warning('Errore durante la simulazione');
        speed_gain = -1e6; % Penalità in caso di errore
        disp(ME);
    end

    % Chiudi il modello senza salvare modifiche
    close_system(model_name, 0);
end