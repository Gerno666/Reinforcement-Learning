% Specify the model name
model_name = 'sldemo_autotrans';

% Load the Simulink model
load_system(model_name);

% Set the simulation duration to 300 seconds
set_param(model_name, 'StopTime', '300');
set_param(model_name, 'SignalLogging', 'on');
set_param(model_name, 'SignalLoggingName', 'sldemo_autotrans_output');

% Start the simulation
disp('Starting simulation...');

% Run the simulation and save the output to simOut
simOut = sim(model_name, 'ReturnWorkspaceOutputs', 'on');

% Extract simulation results from logsout
logsout = simOut.sldemo_autotrans_output;

% Extract throttle signal data
try
    throttle_signal = logsout.getElement('Throttle').Values;
    throttle_time = throttle_signal.Time;
    throttle_data = throttle_signal.Data;
catch
    warning('Throttle signal not found in logsout.');
    throttle_time = [];
    throttle_data = [];
end

% Extract brake signal data
try
    brake_signal = logsout.getElement('Brake').Values;
    brake_time = brake_signal.Time;
    brake_data = brake_signal.Data;
catch
    warning('Brake signal not found in logsout.');
    brake_time = [];
    brake_data = [];
end

% Extract vehicle speed signal data
try
    speed_signal = logsout.getElement('VehicleSpeed').Values;
    speed_time = speed_signal.Time;
    speed_data = speed_signal.Data;
catch
    warning('Vehicle speed signal not found in logsout.');
    speed_time = [];
    speed_data = [];
end

% Extract engine RPM signal data
try
    rpm_signal = logsout.getElement('EngineRPM').Values;
    rpm_time = rpm_signal.Time;
    rpm_data = rpm_signal.Data;
catch
    warning('Engine RPM signal not found in logsout.');
    rpm_time = [];
    rpm_data = [];
end

% Calculate KPI violation based on RPM and Speed
kpi_data = zeros(size(rpm_data));  % Initialize KPI as a zero array
for i = 1:length(rpm_data)
    rpm_violation = (rpm_data(i) - 4500) / 4500;  % Negative if below 4500, positive above
    speed_violation = (speed_data(i) - 120) / 120;  % Negative if below 120, positive above
    kpi_data(i) = (rpm_violation + speed_violation) / 2;  % Calculate KPI as the average
end
kpi_time = rpm_time;  % Use RPM time for KPI plot

% Calculate and display the average KPI violation over the simulation
if ~isempty(kpi_data)
    average_violation = mean(kpi_data);  % Calculate average KPI violation
    disp(['Average KPI Violation: ', num2str(average_violation)]);
else
    disp('KPI data not available. Unable to calculate average violation.');
    average_violation = NaN;  % Handle case where KPI data is not available
end

% Display and save the simulation results in plots
figure('Units', 'normalized', 'Position', [0.1, 0.1, 0.8, 0.8]); % 80% dello schermo

% Plot throttle signal over time
subplot(5,1,1);
if ~isempty(throttle_data)
    plot(throttle_time, throttle_data);
    title('Throttle Over Time');
    xlabel('Time (s)');
    ylabel('Throttle (%)');
else
    title('Throttle data not available');
end

% Plot brake signal over time
subplot(6,1,2);
if ~isempty(brake_data)
    plot(brake_time, brake_data);
    title('Brake Over Time');
    xlabel('Time (s)');
    ylabel('Brake (%)');
else
    title('Brake data not available');
end

% Plot vehicle speed over time
subplot(6,1,3);
if ~isempty(speed_data)
    plot(speed_time, speed_data);
    hold on;
    yline(120, 'r--', 'Target Speed'); % Add red dashed line for target speed
    title('Vehicle Speed Over Time');
    xlabel('Time (s)');
    ylabel('Speed (km/h)');
    hold off;
else
    title('Vehicle speed data not available');
end

% Plot engine RPM over time
subplot(6,1,4);
if ~isempty(rpm_data)
    plot(rpm_time, rpm_data);
    hold on;
    yline(4500, 'r--', 'Target RPM'); % Add red dashed line for target RPM
    title('Engine RPM Over Time');
    xlabel('Time (s)');
    ylabel('RPM');
    hold off;
else
    title('Engine RPM data not available');
end

% Plot KPI violation level over time
subplot(6,1,5);
if ~isempty(kpi_data)
    plot(kpi_time, kpi_data);
    hold on;
    yline(average_violation, 'r--', 'Average Violation'); % Add red dashed line for average violation
    title('KPI Violation Over Time');
    xlabel('Time (s)');
    ylabel('KPI');
    hold off;
else
    title('KPI (Violation) data not available');
end

sgtitle('Simulation Results');

% Save plots as a PNG file
saveas(gcf, 'simulation_results.png');

% Close the model without saving changes
close_system(model_name, 0);

disp('Simulation completed and plots saved.');