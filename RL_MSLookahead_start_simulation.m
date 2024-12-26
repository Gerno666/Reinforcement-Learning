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