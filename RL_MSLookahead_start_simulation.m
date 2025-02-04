% Nome del modello Simulink
model_name = 'RL_MSLookahead_sldemo_autotrans';

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

terminate = false;

% Timer per il monitoraggio
monitor_timer = timer('ExecutionMode', 'fixedRate', 'Period', 1, ...
    'TimerFcn', @(~,~) log_resources());

start(monitor_timer); % Avvia il timer di monitoraggio

for i = 1:100
    fprintf('Caricamento del modello Simulink: %s\n', model_name);
    
    % Carica il modello
    if ~bdIsLoaded(model_name)
        load_system(model_name);
        fprintf('Modello %s caricato correttamente.\n', model_name);
    else
        fprintf('Modello %s già caricato.\n', model_name);
    end
    
    % Configura la simulazione
    fprintf('Configurazione della simulazione...\n');
    simIn = Simulink.SimulationInput(model_name);
    stopTime = 100000;
    simIn = simIn.setModelParameter('StopTime', 'stopTime');
    set_param(model_name, 'SignalLogging', 'on');
    fprintf('Simulazione configurata con durata infinita (StopTime = 100000) e logging abilitato.\n');
    
    % Avvia la simulazione
    fprintf('Avvio della simulazione per il modello %s...\n', model_name);
    try
        simOut = sim(simIn);
        fprintf('Simulazione completata correttamente.\n');
        
        % Recupera i segnali registrati
        logsout = simOut.get('sldemo_autotrans_output');
        time = logsout.getElement('time').Values.Data; % Tempo

        if time(end) < stopTime
            terminate = true;
            fprintf('La simulazione è stata interrotta prima dello StopTime.\n');
            break; % Uscita dal ciclo
        end
    
    catch ME
        fprintf('Errore durante la simulazione: %s\n', ME.message);
        terminate = true; % Setta terminate a true in caso di errore
        break; % Uscita dal ciclo
    end
    
    % Chiudi il modello per liberare memoria
    fprintf('Chiusura del modello %s per liberare memoria...\n', model_name);
    close_system(model_name, 0);
end

stop(monitor_timer); % Ferma il timer di monitoraggio
delete(monitor_timer); % Elimina il timer

if terminate
    fprintf('Simulazione interrotta. La policy ha raggiunto la convergenza o si è verificato un errore.\n');
else
    fprintf('Simulazione completata per tutte le iterazioni.\n');
end

% Genera il grafico delle risorse
cpu_normalized = cpu_usage / 8; % Normalizza per il numero di core
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
saveas(gcf, 'CPU_RAM_Monitoring_Simulation.png');
fprintf('Grafico salvato come CPU_RAM_Monitoring_Simulation.png\n');

fprintf('Processo completato. Il modello %s è stato chiuso.\n', model_name);

%{
% Nome del modello Simulink
model_name = 'RL_MSLookahead_sldemo_autotrans';

terminate = false;

for i = 1:100
    fprintf('Caricamento del modello Simulink: %s\n', model_name);
    
    % Carica il modello
    if ~bdIsLoaded(model_name)
        load_system(model_name);
        fprintf('Modello %s caricato correttamente.\n', model_name);
    else
        fprintf('Modello %s già caricato.\n', model_name);
    end
    
    % Configura la simulazione
    fprintf('Configurazione della simulazione...\n');
    simIn = Simulink.SimulationInput(model_name);
    stopTime = 100000;
    simIn = simIn.setModelParameter('StopTime', 'stopTime');
    set_param(model_name, 'SignalLogging', 'on');
    fprintf('Simulazione configurata con durata infinita (StopTime = 100000) e logging abilitato.\n');
    
    % Avvia la simulazione
    fprintf('Avvio della simulazione per il modello %s...\n', model_name);
    try
        simOut = sim(simIn);
        fprintf('Simulazione completata correttamente.\n');
        
        % Recupera i segnali registrati
        % Estrai i segnali dai risultati della simulazione
        logsout = simOut.get('sldemo_autotrans_output');
        
        % Estrai i segnali necessari
        time = logsout.getElement('time').Values.Data; % Tempo

        if time(end) < stopTime
            terminate = true;
            fprintf('La simulazione è stata interrotta prima dello StopTime.\n');
            break; % Uscita dal ciclo
        end
    
    catch ME
        fprintf('Errore durante la simulazione: %s\n', ME.message);
        terminate = true; % Setta terminate a true in caso di errore
        break; % Uscita dal ciclo
    end
    
    % Chiudi il modello per liberare memoria
    fprintf('Chiusura del modello %s per liberare memoria...\n', model_name);
    close_system(model_name, 0);

end

if terminate == true
    fprintf('Simulazione interrotta. La policy ha raggiunto la convergenza o si è verificato un errore.\n');
else
    fprintf('Simulazione completata per tutte le iterazioni.\n');
end

fprintf('Processo completato. Il modello %s è stato chiuso.\n', model_name);

%}