% Carica i dati salvati
filename = 'RL_10k_simulations_data.mat';
if isfile(filename)
    data = load(filename);
    if isfield(data, 'all_data')
        all_data = data.all_data;

        % Rimuovi eventuali righe con NaN
        all_data = all_data(~any(isnan(all_data), 2), :);

        % Definisci input (stati) e output (costi)
        inputs = all_data(:, 1:end-1); % Tutte le colonne tranne l'ultima
        targets = all_data(:, end);    % Ultima colonna (costo)

        % Calcolo manuale di media e deviazione standard
        input_mean = mean(inputs, 1); % Media per ogni colonna (feature)
        input_std = std(inputs, 0, 1); % Deviazione standard per ogni colonna (feature)

        % Correzione: evita divisione per 0 sostituendo deviazioni standard nulle con 1
        input_std(input_std == 0) = 1;

        % Normalizzazione manuale degli input
        inputs_normalized = (inputs - input_mean) ./ input_std;

        % Suddividi i dati in 80% training e 20% test
        num_samples = size(inputs_normalized, 1);
        idx = randperm(num_samples); % Shuffle casuale
        train_idx = idx(1:round(0.8 * num_samples)); % Indici per il training (80%)
        test_idx = idx(round(0.8 * num_samples) + 1:end); % Indici per il test (20%)

        % Dati di training
        train_inputs = inputs_normalized(train_idx, :);
        train_targets = targets(train_idx, :);

        % Dati di test
        test_inputs = inputs_normalized(test_idx, :);
        test_targets = targets(test_idx, :);

        % Parametri della rete neurale
        input_size = size(train_inputs, 2); % Numero di feature dello stato
        hidden_layer_size = 50;            % Numero di neuroni nel layer nascosto
        output_size = 1;                   % Predice un solo valore (costo)

        % Definizione della rete neurale
        net = fitnet(hidden_layer_size, 'trainlm'); % Usa il training Levenberg-Marquardt
        net.trainParam.epochs = 1000;       % Numero massimo di epoche
        net.trainParam.goal = 1e-6;         % Obiettivo (errore minimo)
        net.trainParam.min_grad = 1e-10;    % Gradiente minimo
        net.trainParam.max_fail = 6;        % Numero massimo di errori consecutivi

        % Esegui il training con i dati di training
        [net, tr] = train(net, train_inputs', train_targets');

        % Valutazione delle performance sui dati di training
        train_outputs = net(train_inputs');
        train_performance = perform(net, train_targets', train_outputs);
        fprintf('Errore medio quadratico (MSE) sui dati di training: %.4f\n', train_performance);

        % Valutazione della rete sui dati di test
        test_outputs = net(test_inputs');
        test_performance = perform(net, test_targets', test_outputs);
        fprintf('Errore medio quadratico (MSE) sui dati di test: %.4f\n', test_performance);

        % Salva il modello addestrato con i parametri di normalizzazione
        input_norm_params.mean = input_mean;
        input_norm_params.std = input_std;
        save('trained_cost_model.mat', 'net', 'input_norm_params');
        fprintf('Modello addestrato e parametri di normalizzazione salvati in trained_cost_model.mat\n');

        % Plot dei risultati di test
        figure;
        plot(test_targets, 'r', 'DisplayName', 'Costo Reale');
        hold on;
        plot(test_outputs, 'b--', 'DisplayName', 'Costo Predetto');
        xlabel('Campioni');
        ylabel('Costo');
        title('Confronto tra costi reali e predetti sui dati di test');
        legend('show');
        hold off;
        saveas(gcf, 'test_results.png');
        fprintf('Risultati del test salvati in test_results.png\n');
    else
        error('La variabile "all_data" non è presente nel file.');
    end
else
    error('Il file RL_simulation_data.mat non esiste.');
end