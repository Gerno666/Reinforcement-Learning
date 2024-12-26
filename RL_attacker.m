function [throttle, brake] = RL_attacker(sliding_window, weights, bias_input, weights_hidden, bias_output)
    % Genera valori di throttle e brake utilizzando i pesi e i bias salvati della rete.
    % 
    % Args:
    %   sliding_window (1x300 double): Finestra temporale di input.
    %   weights (matrix): Pesi del primo layer.
    %   bias_input (vector): Bias del primo layer.
    %   weights_hidden (matrix): Pesi del secondo layer.
    %   bias_output (vector): Bias del secondo layer.
    %
    % Returns:
    %   throttle (double): Valore predetto di throttle.
    %   brake (double): Valore predetto di brake.

    % Parametri di media e covarianza
    THROTTLE_MEAN = 50;
    BRAKE_MEAN = 500;
    COVARIANCE_MATRIX = [30, -15; -15, 400];

    % Assicura che sliding_window abbia la forma corretta
    sliding_window = reshape(sliding_window, 1, []); % Riorganizza in vettore riga

    % Feedforward attraverso la rete
    hidden_layer = max(0, sliding_window * weights' + bias_input'); % ReLU
    output_layer = hidden_layer * weights_hidden' + bias_output'; % Output lineare

    % Estrai i valori predetti di throttle e brake
    predicted_throttle = output_layer(1);
    predicted_brake = output_layer(2);

    % Genera nuovi valori di throttle e brake rispettando media e covarianza
    adjusted_inputs = mvnrnd([THROTTLE_MEAN, BRAKE_MEAN], COVARIANCE_MATRIX, 1);

    % Applica i vincoli di throttle e brake
    if predicted_throttle > predicted_brake
        throttle = max(0, round(adjusted_inputs(1)));
        brake = 0; % Assicura che throttle > 0 implica brake = 0
    else
        throttle = 0;
        brake = max(0, round(adjusted_inputs(2))); % Assicura che brake > 0 implica throttle = 0
    end
end