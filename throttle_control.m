function [throttle, brake] = throttle_control(EngineRPM, VehicleSpeed, time)
    % Define Python functions as extrinsic so they can be called from Simulink
    coder.extrinsic('py.numpy.array');
    coder.extrinsic('py.optimization_final.calculate_optimal_controls');

    % Set the random seed based on the current time for true randomness
    rng('shuffle');

    % Initialize persistent variables to store the last 10 values of Speed, RPM, and Remaining Fuel
    persistent speed_values rpm_values
    if isempty(speed_values)
        speed_values = zeros(1, 10);  % Initialize speed values to zero
        rpm_values = zeros(1, 10);    % Initialize RPM values to zero
    end

    % Update the last 10 values for speed, RPM, and Remaining Fuel
    speed_values = [speed_values(2:end), VehicleSpeed];
    rpm_values = [rpm_values(2:end), EngineRPM];

    % Define normalization constants (update these based on system limits)
    max_speed = 163;  % Maximum possible speed in km/h
    max_rpm = 5030;   % Maximum possible engine RPM

    % Normalize the state values
    norm_speed = speed_values / max_speed;
    norm_rpm = rpm_values / max_rpm;

    % For the first 10 seconds, generate random throttle and brake values
    if time < 11
        throttle = round(rand() * 100);  % Random throttle between 0 and 100
        brake = round(rand() * 2500);   % Random brake between 0 and 2500
    else
        % After the first 10 seconds, combine state and call the Python function
        % Combine normalized state values
        state = [norm_speed, norm_rpm];

        py_state = py.numpy.array(state(:));  % Convert the state to a Python array

        % Call the Python function to get optimal throttle and brake values
        temp_result = py.optimization_final.calculate_optimal_controls(py_state);
        
        % Convert the result to a double array
        temp_result_double = double(temp_result);
        
        % Extract throttle and brake values from the result and apply limits
        throttle = max(0, min(temp_result_double(1), 100));
        brake = max(0, min(temp_result_double(2), 2500));
    end

    % Debug information
    fprintf("Time: %.2f s, Throttle: %.2f, Brake: %.2f\n", ...
            time, throttle, brake);
end