% MULTIDIMENSIONAL OPTIMAL TRANSPORT

% RANDOMLY GENERATED 2D POINTS
rng('default');
x = normrnd(0, 0.5, [100,2]);
y = normrnd(2.5, 0.5, [100,2]);

% STARTING PARAMETERS
eta_init = 0.05;
iter_num = 100;
iters = 1:iter_num;

%RUNNING GRADIENT DESCENT
[T_hist, L1_hist, L2_hist, L_hist, eta_hist] = grad_descent(x, y, eta_init, iter_num);
%DO MORE ITERATIONS WITH FINAL LAMBDA

%PLOTTING INITIAL DISTRIBUTION
figure()
hold on
scatter(x(:,1), x(:,2), 'filled', color = 'blue')
scatter(y(:,1), y(:,2), 'filled', color = 'red')
hold off

figure()
hold on
%PLOT C(X, TX)
subplot(2, 2, 1)
plot(iters, L1_hist, '-')
title('L1 Cost')

%PLOT (F1 - F2)
subplot(2, 2, 2)
plot(iters, L2_hist, '-')
title('L2 Cost')

%PLOT GLOBAL COST
subplot(2, 2, 3)
plot(iters, L_hist, '-')
title('L Cost')

%PLOT LEARNING RATE/STEP SIZE ETA
subplot(2, 2, 4)
plot(iters, eta_hist, '-')
title('Eta')
hold off

% PLOT TRAJECTORY OF EACH POINT
figure()
hold on
T_map = T_hist(:,:,iter_num+1);
for i = 1:iter_num
    T_map_i1 = T_hist(:,:,i);
    T_map_i2 = T_hist(:,:,i+1);
    for j = 1:length(x)
        plot([T_map_i1(j,1) T_map_i2(j,1)], [T_map_i1(j,2) T_map_i2(j,2)], color = 'green')
    end
end

% PLOTTING FINAL OPTIMAL MAP
scatter(x(:,1), x(:,2), 'filled', color = 'blue')
scatter(y(:,1), y(:,2), 'filled', color = 'red')
scatter(T_map(:,1), T_map(:,2), 'filled', color = 'green')
title('Final Map')
hold off

% BANDWIDTH MATRIX SELECTION WITH SILVERMAN'S RULE OF THUMB
function H = bandwidth(x, n)
    [~, d] = size(x);
    H = diag(diag(cov(x)))*(4/((d+2)*n))^(1/(d+4));
end

% COST FUNCTION C (RETURNS CONSTANT)
function cost = C(x, Tx)
    cost = (norm(x - Tx, 'fro')^2) / (2*length(x));
end

% TEST FUNCTION PART F1 (RETURNS CONSTANT)
function f1 = F1(Tx, y, Hx, Hy)
    n = length(Tx);
    m = length(y);
    func1 = 0;
    func2 = 0;
    for i = 1:n
        for j = 1:n
            func1 = func1 + mvnpdf(Tx(i,:), Tx(j,:), Hx);
        end
        for k = 1:m
            func2 = func2 + mvnpdf(Tx(i,:), y(k,:), Hy);
        end
    end
    f1 = (func1/(n^2)) - (func2/(m*n));
end

% TEST FUNCTION PART F2 (RETURNS CONSTANT)
function f2 = F2(Tx, y, Hx, Hy)
    n = length(Tx);
    m = length(y);
    func1 = 0;
    func2 = 0;
    for i = 1:m
        for j = 1:n
            func1 = func1 + mvnpdf(y(i,:), Tx(j,:), Hx);
        end
        for k = 1:m
            func2 = func2 + mvnpdf(y(i,:), y(k,:), Hy);
        end
    end
    f2 = (func1/(m*n)) - (func2/(m^2));
end

% GLOBAL COST FUNCTION L (RETURNS CONSTANT)
function l = L(x, y, Tx, Hx, Hy, lam)
    l = C(x, Tx) + lam*(F1(Tx, y, Hx, Hy) - F2(Tx, y, Hx, Hy));
end

% GRADIENT OF COST (RETURNS 1xD VECTOR)
function c_grad_i = C_grad_i(x_i, Tx_i)
    c_grad_i = (Tx_i - x_i) ./ length(x_i);
end

% GRADIENT OF F1 (RETURNS 1xD VECTOR)
function f_grad_i = F_grad_i(Tx, y, Tx_i, Hx, Hy)
    [n, d] = size(Tx);
    m = length(y);
    func1 = zeros(1,d);
    func2 = zeros(1,d);
    func3 = zeros(1,d);
    for j = 1:n
        func1 = func1 + (mvnpdf(Tx_i, Tx(j,:), Hx) .* (Hx\(Tx(j,:) - Tx_i).').');
    end
    for k = 1:m
        func2 = func2 + (mvnpdf(Tx_i, y(k,:), Hy) .* (Hy\(y(k,:) - Tx_i).').');
        func3 = func3 + (mvnpdf(y(k,:), Tx_i, Hx) .* (Hx\(y(k,:) - Tx_i).').');
    end
    f1_grad_i = (func1/(n^2)) - (func2/(m*n));
    f2_grad_i = func3/(n*m);
    f_grad_i = f1_grad_i - f2_grad_i;
end

% GRADIENT OF L (RETURNS NxD MATRIX)
function l_grad = L_grad(x, y, Tx, Hx, Hy, lam)
    [n, d] = size(x);
    l_grad = zeros(n,d);  % INITIALIZE L GRADIENT TO ZEROS
    for i = 1:n           % ADD VALUE TO L GRADIENT MATRIX AT EACH I
        l_grad(i,:) = C_grad_i(x(i,:), Tx(i,:)) + lam*(F_grad_i(Tx, y, Tx(i,:), Hx, Hy));
    end
end

% ADAPTIVE LEARNING RATE ETA (RETURNS CONSTANT AND GRAD DESCENT RESULT)
function [eta, Tx_next] = adapt_learning(x, y, Tx_curr, Hx, Hy, lam, eta)
    eta = eta * 2;                                                    % INCREASE ETA FOR FASTER CONVERGENCE
    Tx_next = Tx_curr - (eta .* L_grad(x, y, Tx_curr, Hx, Hy, lam));  % COMPUTE NEW MAP TX
    L_curr = L(x, y, Tx_curr, Hx, Hy, lam);                           % COMPUTE COST BASED ON PAST MAP
    L_next = L(x, y, Tx_next, Hx, Hy, lam);                           % COMPUTE COST BASED ON NEW MAP
    max_while = 5;
    while L_curr < L_next % NEXT COST L SHOULD NOT BE GREATER THAN THE CURRENT ONE
        if max_while < 0 % MAX LENGTH OF LOOP SHOULD NOT EXCEED 5 OR ETA WILL BECOME TOO SMALL
            break
        end
        eta = eta / 2;                                                    % SHRINK STEP SIZE
        Tx_next = Tx_curr - (eta .* L_grad(x, y, Tx_curr, Hx, Hy, lam));  % COMPUTE NEW MAP TX WITH NEW ETA
        L_next = L(x, y, Tx_next, Hx, Hy, lam);                           % COMPUTE COST BASED ON NEW MAP
        max_while = max_while - 1;                                        % KEEP TRACK OF WHILE-LOOP LENGTH
    end
end

% LINEARLY INCREASING LAMBDA AND DECREASING BANDWIDTH (RETURNS DxD MATRIX)
function [Hz_new, lam_new] = linear_change(Hy, Hz, i, it, H_const, lam_init, lam_final)
    Hz_new = (H_const * Hz * (it - i) / it) + (Hy * i / it);
    lam_new = (lam_init * (it - i) / it) + (lam_final * i / it); 
end

% GRADIENT DESCENT (MAYBE USE SOME FIXED POINT ITERATION TO GET THE IMPLICIT VERSION)
function [T_hist, L1_hist, L2_hist, L_hist, eta_hist] = grad_descent(x, y, eta, it)

    % INITIALIZING HISTORY TO ZEROS FOR PLOTTING OVER TIME
    T_hist = x; % INITIAL MAP SHOULD BE ORIGINAL SET OF POINTS
    L1_hist = zeros(it, 1);
    L2_hist = zeros(it, 1);
    L_hist = zeros(it, 1);
    eta_hist = zeros(it, 1);

    % INITIAL VALUES
    z = [x; y];                         % COMBINED SET OF POINTS, FOR BANDWIDTH
    Hy = bandwidth(y, length(z));       % BANDWIDTH FOR Y
    Hz_init = bandwidth(z, length(z));  % BANDWIDTH FOR ALL POINTS
    Tx = x;                             % INITIAL MAP SHOULD BE THE ORIGINAL SET OF POINTS

    % LOOP OVER ITERATIONS
    for i = 1:it
        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(i, 10) == 0
            fprintf("Iteration: %d\n", i)
        end

        H_const = 9;           % MULTIPLY BANDWIDTH BY THIS FACTOR TO REACH ALL POINTS
        lambda_init = 5000;    % INITIAL REGULARIZATION PARAMETER
        lambda_final = 50000;  % FINAL REGULARIZATION PARAMETER (SHOULD ALWAYS INCREASE)

        % GETTING NEW BANDWIDTH (DECREASE TO HY) AND LAMBDA (INCREASE TO FINAL)
        [Hz, lam] = linear_change(Hy, Hz_init, i, it, H_const, lambda_init, lambda_final);

        % GET NEW MAP TX AND LEARNING RATE ETA AT EACH STEP
        [eta, Tx] = adapt_learning(x, y, Tx, Hz, Hz, lam, eta);

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        eta_hist(i,:) = eta;
        T_hist(:,:,i+1) = Tx;
        L1_hist(i,:) = C(x, Tx);
        L2_hist(i,:) = F1(y, Tx, Hz, Hz) - F2(y, Tx, Hz, Hz);
        L_hist(i,:) = C(x, Tx) + lam*(F1(y, Tx, Hz, Hz) - F2(y, Tx, Hz, Hz));
    end
       
    disp("Running additional iters")
    for i = 1:20 % ADDITIONAL ITERATIONS WITH THE FINAL BANDWIDTH AND LAMBDA
        [eta, Tx] = adapt_learning(x, y, Tx, Hz, Hz, lam, eta);
        if mod(i, 10) == 0
            fprintf("Iteration: %d\n", i)
        end

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        eta_hist(i,:) = eta;
        T_hist(:,:,i+1) = Tx;
        L1_hist(i,:) = C(x, Tx);
        L2_hist(i,:) = F1(y, Tx, Hz, Hz) - F2(y, Tx, Hz, Hz);
        L_hist(i,:) = C(x, Tx) + lam*(F1(y, Tx, Hz, Hz) - F2(y, Tx, Hz, Hz));
    end

end