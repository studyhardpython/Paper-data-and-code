clear; clc; close all;

porosity = [2, 3, 4, 5, 6];
diameter = [16, 30, 48, 64, 80];
aspect_ratio = [0, 2, 3];

data_folder = 'processed_data';

all_combos = {};
for p = porosity
    for d = diameter
        for ar = aspect_ratio
            all_combos{end+1} = [p, d, ar];
        end
    end
end

num_combos = length(all_combos);   % 共5×5×3 = 75组基准参数组合

rng(42);
combo_idx = randperm(num_combos);

n_train_combos = 60;               % 随机选取60组用于训练
n_test_combos = 15;                % 剩余15组用于测试

train_combo_idx = combo_idx(1:n_train_combos);
test_combo_idx = combo_idx(n_train_combos+1 : n_train_combos+n_test_combos);

X_train = []; Y_train = [];
X_test = []; Y_test = [];

for k = train_combo_idx
    combo = all_combos{k};
    p = combo(1); d = combo(2); ar = combo(3);
    filename = sprintf('%s/processed_%d_%d_%d.mat', data_folder, p, d, ar);
    data = load(filename);
    C_selected = data.C_selected;
    H_selected = data.H_selected;
    num_samples = size(C_selected, 3);
    if ar == 0
        inputs = repmat([p/10, d, ar+1], num_samples, 1);
    else
        inputs = repmat([p/10, d, ar], num_samples, 1);
    end
    outputs = zeros(num_samples, 6);
    for i = 1:num_samples
        outputs(i, 1) = C_selected(1, 1, i);
        outputs(i, 2) = C_selected(1, 2, i);
        outputs(i, 3) = C_selected(2, 2, i);
        outputs(i, 4) = C_selected(3, 3, i);
        outputs(i, 5) = H_selected(1, 1, i);
        outputs(i, 6) = H_selected(2, 2, i);
    end
    X_train = [X_train; inputs];
    Y_train = [Y_train; outputs];
end

for k = test_combo_idx
    combo = all_combos{k};
    p = combo(1); d = combo(2); ar = combo(3);
    filename = sprintf('%s/processed_%d_%d_%d.mat', data_folder, p, d, ar);
    data = load(filename);
    C_selected = data.C_selected;
    H_selected = data.H_selected;
    num_samples = size(C_selected, 3);
    if ar == 0
        inputs = repmat([p/10, d, ar+1], num_samples, 1);
    else
        inputs = repmat([p/10, d, ar], num_samples, 1);
    end
    outputs = zeros(num_samples, 6);
    for i = 1:num_samples
        outputs(i, 1) = C_selected(1, 1, i);
        outputs(i, 2) = C_selected(1, 2, i);
        outputs(i, 3) = C_selected(2, 2, i);
        outputs(i, 4) = C_selected(3, 3, i);
        outputs(i, 5) = H_selected(1, 1, i);
        outputs(i, 6) = H_selected(2, 2, i);
    end
    X_test = [X_test; inputs];
    Y_test = [Y_test; outputs];
end

input_min = min([X_train; X_test], [], 1);
input_max = max([X_train; X_test], [], 1);
output_min = min([Y_train; Y_test], [], 1);
output_max = max([Y_train; Y_test], [], 1);

X_train_norm = ((X_train - input_min) ./ (input_max - input_min))';
Y_train_norm = ((Y_train - output_min) ./ (output_max - output_min))';
X_test_norm = ((X_test - input_min) ./ (input_max - input_min))';
Y_test_norm = ((Y_test - output_min) ./ (output_max - output_min))';

hidden_layer1_size = 64;
hidden_layer2_size = 32;

net = fitnet([hidden_layer1_size, hidden_layer2_size], 'trainlm');

net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.trainParam.min_grad = 1e-10;
net.trainParam.mu = 0.001;
net.trainParam.mu_dec = 0.1;
net.trainParam.mu_inc = 10;
net.trainParam.mu_max = 1e10;
net.trainParam.showWindow = true;

net.performFcn = 'mse';

[net, tr] = train(net, X_train_norm, Y_train_norm);

Y_test_pred = net(X_test_norm);

output_names = {'c11', 'c12', 'c22', 'c33', 'k11', 'k22'};
for i = 1:6
    r2_i = 1 - sum((Y_test_norm(i,:) - Y_test_pred(i,:)).^2) / ...
               sum((Y_test_norm(i,:) - mean(Y_test_norm(i,:))).^2);
    fprintf('%s: R2 = %.4f\n', output_names{i}, r2_i);
end

save('trained_model.mat', 'net', 'tr', 'input_min', 'input_max', 'output_min', 'output_max');