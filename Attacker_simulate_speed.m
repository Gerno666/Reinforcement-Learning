function [delta_speed] = Attacker_simulate_speed(throttle, brake)
    % Simula il guadagno di velocità basato sulle coppie di throttle e brake.
    % Args:
    %   throttle (double array): Valori di throttle [30 x 1].
    %   brake (double array): Valori di brake [30 x 1].
    % Returns:
    %   delta_speed (double): Differenza di velocità tra la velocità finale e la velocità intermedia dopo 20 coppie.

    % Verifica le dimensioni degli input
    if length(throttle) ~= 30 || length(brake) ~= 30
        error('Throttle e Brake devono avere 30 elementi ciascuno.');
    end

    % Nome del modello Simulink
    model_name = 'sldemo_autotrans_test';

    % Carica il modello
    load_system(model_name);

    % Configura i parametri di simulazione
    set_param(model_name, 'StopTime', '30'); % Tempo massimo di simulazione (30 secondi)

    % Configura i dati
    time_generated = linspace(0, 30, 30)'; % Tempo da 0 a 30 secondi
    throttle_generated = throttle(:); % Assicurati che throttle sia una colonna
    brake_generated = brake(:); % Assicurati che brake sia una colonna

    % Crea input completo
    external_input = [time_generated, throttle_generated, brake_generated];

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

        % Calcola la velocità intermedia e finale
        intermediate_speed = speed_signal(20); % Velocità dopo 20 coppie (indice 20)
        final_speed = speed_signal(end); % Velocità finale (indice 30)

        % Calcola il delta speed
        delta_speed = final_speed - intermediate_speed;

    catch ME
        % Gestisci gli errori della simulazione
        warning('Errore durante la simulazione.');
        delta_speed = -1e6; % Penalità in caso di errore
        disp(ME);
    end

    % Chiudi il modello senza salvare modifiche
    close_system(model_name, 0);
    
end