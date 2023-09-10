% Data dumy
%X = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2]; %% input yang diinginkan

%Y = [1; 1; 1; 1; 1; 1; 1; 1; 1; 1; 2; 2; 2; 2; 2; 2; 2; 2; 2; 2]; %% input yang dikeluarkan

% file ujicoba dataset
a = xlsread("coba.xlsx", "Sheet1");
X = a(:,1:4);
Y = a(:,5);
% Menghitung matriks kernel (produk dalam)
n = size(X, 1);
K = X * X';

% Parameter SVM
C = 1; % Parameter C (Cost)
alpha = zeros(n, 1);
tolerance = 1e-5;
maxIterations = 100;

for iteration = 1:maxIterations
    for i = 1:n
        E_i = sum(alpha .* Y .* K(:, i)) - Y(i);
        if (Y(i) * E_i < -tolerance && alpha(i) < C) || (Y(i) * E_i > tolerance && alpha(i) > 0)
            j = randi([1, n], 1);
            while j == i
                j = randi([1, n], 1);
            end
            E_j = sum(alpha .* Y .* K(:, j)) - Y(j);
            
            % Simpan nilai lama alpha
            alpha_i_old = alpha(i);
            alpha_j_old = alpha(j);
            
            % Hitung batas atas dan bawah
            if Y(i) == Y(j)
                L = max(0, alpha_i_old + alpha_j_old - C);
                H = min(C, alpha_i_old + alpha_j_old);
            else
                L = max(0, alpha_j_old - alpha_i_old);
                H = min(C, C + alpha_j_old - alpha_i_old);
            end
            
            if L == H
                continue;
            end
            
            eta = 2 * K(i, j) - K(i, i) - K(j, j);
            if eta >= 0
                continue;
            end
            
            alpha(j) = alpha_j_old - Y(j) * (E_i - E_j) / eta;
            alpha(j) = min(H, alpha(j));
            alpha(j) = max(L, alpha(j));
            
            alpha(i) = alpha_i_old + Y(i) * Y(j) * (alpha_j_old - alpha(j));
        end
    end
end

% Menentukan vektor bobot
w = sum(repmat(alpha .* Y, 1, size(X, 2)) .* X);

% Menentukan vektor support (yang memiliki alpha > 0)
supportVectors = X(alpha > 0, :);

% Menentukan b (intercept)
b = mean(Y - X * w');

% Menampilkan hasil
Y_pred = sign(X * w' + b);

% ========================== perulangan b = 2.4728e+09

disp('Vektor Bobot (w):');
disp(w);
disp('Vektor Support:');
disp(supportVectors);
disp('Nilai b (Intercept):');
disp(b);

count = 0;

for i = 1:length(Y)
    if Y_pred(i,1) == -1
        Y_pred(i,1) = 2;
    else 
        Y_pred(i,1) = 1;
    end
end

for i = 1:length(Y)
    if Y(i,1) == Y_pred(i,1)
        count = count + 1;
    end
end

presentase_prediksi = (count/(length(Y))) * 100


