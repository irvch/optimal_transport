% MULTIDIMENSIONAL OPTIMAL TRANSPORT

% SYNTHETIC DATA IN THE SHAPE OF A GRID 
X1 = [];
X2 = [];
for x = 0:4
    for y = 0:4
        X1 = [X1; (x-2)];
        X2 = [X2; (y-2)];
    end
end
x = [X1 X2];

% TARGET POINTS ARE A SIMPLE TRANSLATION OF SOURCE POINTS
y = x + 3;

% STARTING PARAMETERS
eta_init = 0.05;
iter_num = 100;
extra = 0;
total = iter_num + extra;
iters = 1:total;

% START TIMER FOR ALGORITHM
tic

% RUNNING GRADIENT DESCENT
[Ts, L1s, L2s, Ls, etas] = grad_descent(x, y, eta_init, iter_num, extra);
[T_hist, L1_hist, L2_hist, L_hist, eta_hist] = more_iters(x, y, Ts, L1s, L2s, Ls, etas, iter_num, extra);

% MAP RUNTIME
runtime = toc;

% START TIMER FOR PLOTTING
tic

% PLOTTING INITIAL DISTRIBUTION
figure()
hold on
scatter(x(:,1), x(:,2), 'filled', color = 'blue')
scatter(y(:,1), y(:,2), 'filled', color = 'red')
hold off

figure()
hold on
% PLOT C(X, TX)
subplot(2, 2, 1)
plot(iters, L1_hist, '-')
title('L1 Cost')

% PLOT (F1 - F2)
subplot(2, 2, 2)
plot(iters, L2_hist, '-')
title('L2 Cost')

% PLOT GLOBAL COST
subplot(2, 2, 3)
plot(iters, L_hist, '-')
title('L Cost')

% PLOT LEARNING RATE/STEP SIZE ETA
subplot(2, 2, 4)
plot(iters, eta_hist, '-')
title('Eta')
hold off

% PLOT 25 MAP LOCATION HISTORIES
figure()
hold on
count = 1;
for i = 1:total
    if mod(i, floor(total/25)) == 0
        T_map = T_hist(:,:,i);
        subplot(5, 5, count)
        hold on
        scatter(x(:,1), x(:,2), 'filled', color = 'blue')
        scatter(y(:,1), y(:,2), 'filled', color = 'red')
        scatter(T_map(:,1), T_map(:,2), 'filled', color = 'green')
        iterations = sprintf('Iterations: %d', i);
        title(iterations)
        hold off
        if count < 25
            count = count + 1;
        end
    end
end
hold off

% PLOT TRAJECTORY OF EACH POINT
figure()
hold on
for i = 1:iter_num
    T_map_i1 = T_hist(:,:,i);
    T_map_i2 = T_hist(:,:,i+1);
    for j = 1:length(x)
        plot([T_map_i1(j,1) T_map_i2(j,1)], [T_map_i1(j,2) T_map_i2(j,2)], color = 'green')
    end
end

% DISPLAY MATRIX OF ERROR BETWEEN TWO DATASETS
error_matrix = error(y, T_map);
disp("Error matrix:")
disp(error_matrix)

% DISPLAY ERROR TO HELP WITH COLORING POINTS
errors = zeros(length(error_matrix), 1);
for i = 1:length(error_matrix)
    errors(i,:) = abs(error_matrix(i, 1)) + abs(error_matrix(i, 2));
    fprintf("i=%d: %d\n", i, abs(error_matrix(i, 1)) + abs(error_matrix(i, 2)))
end

% PLOTTING FINAL OPTIMAL MAP
T_map = T_hist(:,:,iter_num+1);
scatter(x(:,1), x(:,2), 'filled', color = 'blue')
scatter(y(:,1), y(:,2), 'filled', color = 'red')
scatter(T_map(:,1), T_map(:,2), 'filled', color = 'green')
c = errors;
scatter(T_map(:,1), T_map(:,2), [], c, 'filled')
%labels = 1:25;
%labelpoints(T_map(:,1), T_map(:,2),  labels)
colorbar
colormap(flipud(winter))
title('Final Map')
hold off

% END TIMER AND DISPLAY RUNTIME
plotting = toc;
disp(['Algorithm Runtime: ' num2str(runtime) ' seconds'])
disp(['Plotting Runtime: ' num2str(plotting) ' seconds'])
disp(['Total Elapsed Runtime: ' num2str(runtime + plotting) ' seconds'])



% BEGINNING OPTIMAL TRANSPORT ALGORITHM
% BANDWIDTH MATRIX SELECTION WITH SILVERMAN'S RULE OF THUMB
function H = bandwidth(x, n)
    [~, d] = size(x);                               
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        H = 0.9*min(std(x), iqr(x)/1.34)*n^(-1/5);
    else      % FOR MULTIDIMENSIONAL CASE
        H = diag(diag(cov(x)))*(4/((d+2)*n))^(1/(d+4));
    end
end

% MULTIVARIATE GAUSSIAN DISTRIBUTION
function pdf = gaussian(x, x_i, H)
    [~, d] = size(x);
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        pdf = exp(-0.5 .* (x-x_i) / H)^2 / sqrt(2*pi);
    else      % FOR MULTIDIMENSIONAL CASE
        const = sqrt(det(H)*(2*pi)^d);
        pdf = exp(-0.5 .* (x-x_i) / H * (x - x_i).') / const;
    end
end

% CREATE WEIGHT MATRIX
function [weight, weight_sum] = W(x_i)
    weight = x_i;

end

% COST FUNCTION C (RETURNS CONSTANT)
function cost = C(x, Tx, w)
    [~, d] = size(x);
    gamma = sum(w);
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        cost = gamma*(norm(x - Tx)^2) / 2;
    else      % FOR MULTIDIMENSIONAL CASE
        cost = gamma*(norm(x - Tx, 'fro')^2) / 2;
    end
end

% TEST FUNCTION PART F1 (RETURNS CONSTANT)
function f1 = F1(Tx, y, Hx, Hy)
    n = length(Tx);
    m = length(y);
    func1 = 0;
    func2 = 0;
    for i = 1:n
        for j = 1:n
            func1 = func1 + gaussian(Tx(i,:), Tx(j,:), Hx);
        end
        for k = 1:m
            func2 = func2 + gaussian(Tx(i,:), y(k,:), Hy);
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
            func1 = func1 + gaussian(y(i,:), Tx(j,:), Hx);
        end
        for k = 1:m
            func2 = func2 + gaussian(y(i,:), y(k,:), Hy);
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

% GRADIENT OF TEST FUNCTION WRT TO APPLIED Tx, NOT CENTER Tx (RETURNS 1xD VECTOR)
function f_grad_i = F_grad_i(Tx, y, Tx_i, Hx, Hy)
    [n, d] = size(Tx);
    m = length(y);
    func1 = zeros(1,d);
    func2 = zeros(1,d);
    for j = 1:n
        func1 = func1 + (gaussian(Tx_i, Tx(j,:), Hx) .* (Hx\(Tx(j,:) - Tx_i).').');
    end
    for k = 1:m
        func2 = func2 + (gaussian(Tx_i, y(k,:), Hy) .* (Hy\(y(k,:) - Tx_i).').');
    end
    f_grad_i = (func1/(n^2)) - (func2/(m*n));
end

% GRADIENT OF L (RETURNS NxD MATRIX)
function l_grad = L_grad(x, y, Tx, Hx, Hy, lam)
    [n, d] = size(x);
    l_grad = zeros(n,d);  % INITIALIZE L GRADIENT TO ZEROS
    for i = 1:n           % ADD VALUE TO L GRADIENT MATRIX AT EACH I
        l_grad(i,:) = C_grad_i(x(i,:), Tx(i,:)) + lam*(F_grad_i(Tx, y, Tx(i,:), Hx, Hy));
    end
end

% ERROR MATRIX BETWEEN DATASETS
function err_matrix = error(y, Tx)
    [n, d] = size(Tx);
    z = [Tx; y];                        % COMBINED SET OF POINTS, FOR BANDWIDTH
    Hy = bandwidth(y, length(z));       % BANDWIDTH FOR Y
    err_matrix = zeros(n,d);
    for i = 1:length(Tx)
        err_matrix(i,:) = F_grad_i(Tx, y, Tx(i,:), Hy, Hy);
    end
end

% ADAPTIVE LEARNING RATE ETA (RETURNS CONSTANT AND GRAD DESCENT RESULT)
function [eta, Tx_next] = adapt_learning(x, y, Tx_curr, Hx, Hy, lam, eta)
    eta = eta * 2;                                                        % INCREASE ETA FOR FASTER CONVERGENCE
    Tx_next = Tx_curr - (eta .* L_grad(x, y, Tx_curr, Hx, Hy, lam));      % COMPUTE NEW MAP TX
    L_curr = L(x, y, Tx_curr, Hx, Hy, lam);                               % COMPUTE COST BASED ON PAST MAP
    L_next = L(x, y, Tx_next, Hx, Hy, lam);                               % COMPUTE COST BASED ON NEW MAP
    max_while = 5;
    while L_curr < L_next % NEXT COST L SHOULD NOT BE GREATER THAN THE CURRENT ONE
        if max_while < 0  % MAX LENGTH OF LOOP SHOULD NOT EXCEED 5 OR ETA WILL BECOME TOO SMALL
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
function [T_hist, L1_hist, L2_hist, L_hist, eta_hist] = grad_descent(x, y, eta, iter_num, extra)

    % INITIALIZING HISTORY TO ZEROS FOR PLOTTING OVER TIME
    T_hist = x; % INITIAL MAP SHOULD BE ORIGINAL SET OF POINTS
    eta_hist = zeros(iter_num+extra, 1);
    L1_hist = zeros(iter_num+extra, 1);
    L2_hist = zeros(iter_num+extra, 1);
    L_hist = zeros(iter_num+extra, 1);

    % INITIAL VALUES
    z = [x; y];                         % COMBINED SET OF POINTS, FOR BANDWIDTH
    Hy = bandwidth(y, length(z));       % BANDWIDTH FOR Y
    Hz_init = bandwidth(z, length(z));  % BANDWIDTH FOR ALL POINTS
    Tx = x;                             % INITIAL MAP SHOULD BE THE ORIGINAL SET OF POINTS

    % LOOP OVER ITERATIONS
    for i = 1:iter_num
        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(i, 10) == 0
            fprintf("Iteration: %d\n", i)
        end

        H_const = 50;           % MULTIPLY BANDWIDTH BY THIS FACTOR TO REACH ALL POINTS
        lambda_init = 25000;    % INITIAL REGULARIZATION PARAMETER
        lambda_final = 100000;  % FINAL REGULARIZATION PARAMETER (SHOULD ALWAYS INCREASE)

        % GETTING NEW BANDWIDTH (DECREASE TO HY) AND LAMBDA (INCREASE TO FINAL)
        [Hz, lam] = linear_change(Hy, Hz_init, i, iter_num, H_const, lambda_init, lambda_final);

        % GET NEW MAP TX AND LEARNING RATE ETA AT EACH STEP
        [eta, Tx] = adapt_learning(x, y, Tx, Hz, Hz, lam, eta);

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        T_hist(:,:,i+1) = Tx;
        eta_hist(i,:) = eta;
        L1_hist(i,:) = C(x, Tx);
        L2_hist(i,:) = F1(y, Tx, Hz, Hz) - F2(y, Tx, Hz, Hz);
        L_hist(i,:) = C(x, Tx) + lam*(F1(y, Tx, Hz, Hz) - F2(y, Tx, Hz, Hz));
    end
end

function [T_hist, L1_hist, L2_hist, L_hist, eta_hist] = more_iters(x, y, T_hist, L1_hist, L2_hist, L_hist, eta_hist, iter_num, extra)
    if extra ~= 0
        disp("Running additional iters")
        z = [x; y];
        Hy = bandwidth(y, length(z));
        lam = 50000;
        eta = eta_hist(iter_num,:);
        Tx = T_hist(:,:,iter_num+1);
        for i = 1:extra % ADDITIONAL ITERATIONS WITH THE FINAL BANDWIDTH AND LAMBDA
            % FOR KEEPING TRACK OF ITERATION PROGRESS
            if mod(i, 10) == 0
                fprintf("Iteration: %d\n", i)
            end
            [eta, Tx] = adapt_learning(x, y, Tx, Hy, Hy, lam, eta);
    
            % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
            T_hist(:,:,i+iter_num+1) = Tx;
            eta_hist(i+iter_num,:) = eta;
            L1_hist(i+iter_num,:) = C(x, Tx);
            L2_hist(i+iter_num,:) = F1(y, Tx, Hy, Hy) - F2(y, Tx, Hy, Hy);
            L_hist(i+iter_num,:) = C(x, Tx) + lam*(F1(y, Tx, Hy, Hy) - F2(y, Tx, Hy, Hy));
        end
        % PLOTTING FINAL MAP AFTER ADDITIONAL ITERATIONS
        % PLOT TRAJECTORY OF EACH POINT
        figure()
        hold on
        for i = 1:total
            T_map_i1 = T_hist(:,:,i);
            T_map_i2 = T_hist(:,:,i+1);
            for j = 1:length(x)
                plot([T_map_i1(j,1) T_map_i2(j,1)], [T_map_i1(j,2) T_map_i2(j,2)], color = 'green')
            end
        end
        T_map = T_hist(:,:,total+1);
        scatter(x(:,1), x(:,2), 'filled', color = 'blue')
        scatter(y(:,1), y(:,2), 'filled', color = 'red')
        scatter(T_map(:,1), T_map(:,2), 'filled', color = 'green')
        title('Additional Iters Map')
        hold off
    end
end