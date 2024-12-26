function average_violation = run_simulation_in_simulink(throttle, brake)

    % Display throttle, brake, and remaining fuel values before starting the simulation
    disp(['Throttle: ', num2str(throttle)]);
    disp(['Brake: ', num2str(brake)]);
    
    % Load the Simulink model
    model_name = 'sldemo_autotrans_test';  % Change this to your model name
    load_system(model_name);

    % Set the simulation time to 5 minutes
    set_param(model_name, 'StopTime', '300');

    % Configure input values for throttle, brake, and remaining fuel
    time = [0; 300];  % Start and end time for the simulation
    throttle_data = [throttle; throttle];  % Keep throttle constant
    brake_data = [brake; brake];  % Keep brake constant

    % Format the data for Simulink input (time, throttle, brake)
    external_input = [time, throttle_data, brake_data];

    % Configure external input in Simulink.SimulationInput
    in = Simulink.SimulationInput(model_name);
    in = in.setExternalInput(mat2str(external_input));

    % Enable signal logging
    set_param(model_name, 'SignalLogging', 'on');
    set_param(model_name, 'SignalLoggingName', 'sldemo_autotrans_output');

    % Run the simulation
    try
        sim_out = sim(in);
    catch
        warning('Simulation failed or was interrupted.');
        average_violation = NaN;  % Return NaN to indicate failure
        return;
    end

    % Extract speed, RPM, and remaining fuel signals from the logs
    logsout = sim_out.sldemo_autotrans_output;
    speed_signal = logsout.getElement('VehicleSpeed').Values;
    rpm_signal = logsout.getElement('EngineRPM').Values;
    
    speed_data = speed_signal.Data;
    rpm_data = rpm_signal.Data;

    % Define target values for speed and RPM
    target_speed = 120;  % Target speed in km/h
    target_rpm = 4500;   % Target RPM

    % Calculate violations as relative differences from target values
    speed_violation = (speed_data - target_speed) / target_speed;
    rpm_violation = (rpm_data - target_rpm) / target_rpm;

    % Compute the average violation over the simulation period
    average_violation = mean((speed_violation + rpm_violation) / 2);

    % Display the average violation value and remaining fuel at the end of the simulation
    disp(['Average Violation: ', num2str(average_violation)]);

    % Close the Simulink model without saving changes
    close_system(model_name, 0);

    % Clear unnecessary variables to free up memory
    clearvars -except average_violation;

    % Clear signal logs
    clear logsout sim_out speed_signal rpm_signal;
end