% Langkah 1: Persiapkan Dataset
load fisheriris
X = meas;         % Fitur-fitur dari dataset
Y = species;      % Kelas target

% Langkah 2: Bagi Data Menjadi Data Latih dan Data Uji
rng(1); % Untuk hasil yang deterministik
cv = cvpartition(Y,'HoldOut',0.3); % Membagi dataset menjadi 70% data latih dan 30% data uji

X_train = X(training(cv),:); % Data latih
Y_train = Y(training(cv));   % Kelas target data latih
X_test = X(test(cv),:);      % Data uji
Y_test = Y(test(cv));        % Kelas target data uji
%%%%%%%%%%%%%%%%
unique_classes = unique(Y_train); % Ambil kelas unik
num_classes = numel(unique_classes);

Y_numeric = zeros(size(Y_train)); % Inisialisasi vektor numerik
for i = 1:num_classes
    Y_numeric(strcmp(Y_train, unique_classes{i})) = i;
end

Y_train = Y_numeric;

unique_classes = unique(Y_test); % Ambil kelas unik
num_classes = numel(unique_classes);

Y_numeric = zeros(size(Y_test)); % Inisialisasi vektor numerik
for i = 1:num_classes
    Y_numeric(strcmp(Y_test, unique_classes{i})) = i;
end
Y_test = Y_numeric;
%%%%%%%%%%%%%%%%
% Langkah 3: Hitung Kernel SVM
K = X_train * X_train'; % Matriks kernel linier

% Langkah 4: Hitung Hessian Matrix
H = (Y_train * Y_train') .* K;

% Langkah 5: Hitung vektor Alpha
f = -ones(size(Y_train));
Aeq = Y_train';
beq = 0;
lb = zeros(size(Y_train));
ub = [];
alpha = quadprog(H, f, [], [], Aeq, beq, lb, ub);

% Langkah 6: Hitung Bobot dan Bias
w = (alpha .* Y_train)' * X_train;
C = 1;
idx = find(alpha >= 0 & alpha < C); % C adalah parameter penalitas
b = Y_train(idx(1)) - w .* X_train(idx(1), :);

% Langkah 7: Lakukan Prediksi pada Data Uji
Y_pred = sign(X_test * w' + b);

