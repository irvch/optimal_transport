% MULTIDIMENSIONAL OPTIMAL TRANSPORT

% RANDOMLY GENERATED 2D POINTS
rng('default');

% GRID SHAPED SOURCE POINTS
%a1 = linspace(-5,0,4);
%a2 = linspace(-5,0,4);
%[A1, A2] = meshgrid(a1, a2);
%x_old = [A1(:) A2(:)];
%x = [A1(:) A2(:)];

% GRID SHAPED TARGET POINTS
%b1 = linspace(1,9,4);
%b2 = linspace(1,9,4);
%[B1, B2] = meshgrid(b1, b2);
%y = [B1(:) B2(:)];

y = table2array(readtable('revised data edit.xlsx', Sheet='every'));
%x = table2array(readtable('revised data edit.xlsx', Sheet='every (2)'));
x = table2array(readtable('revised data set 1.xlsx', Sheet='every (3)'));
%y = table2array(readtable('revised data set 1.xlsx', Sheet='every (4)'));

% PRECONDITIONING
%x1 = (x_old).*std(y)./std(x_old);
%x = x1 - mean(x1) + mean(y);

% STARTING PARAMETERS
eta_init = 0.1;
iter_num = 10;
iters = 1:iter_num;
H_const = 20;          % MULTIPLY BANDWIDTH BY THIS FACTOR TO REACH ALL POINTS
lambda_init = 5000;    % INITIAL REGULARIZATION PARAMETER 
lambda_final = 50000;  % FINAL REGULARIZATION PARAMETER (SHOULD ALWAYS INCREASE)

% START TIMER FOR ALGORITHM
tic

% RUNNING GRADIENT DESCENT
[T_hist, L1_hist, L2_hist, L_hist, eta_hist] = grad_descent(x, y, eta_init, iter_num, H_const, lambda_init, lambda_final);

% MAP RUNTIME
runtime = toc;

% START TIMER FOR PLOTTING
tic

% PLOTTING INITIAL DISTRIBUTIONS
figure()
hold on
scatter3(x(:,1), x(:,2), x(:,3), 'filled', 'blue')
title('SOURCE')
hold off

% PLOTTING DISTRIBUTIONS AFTER PRECONDITIONING
figure()
hold on
scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
title('TARGET')
hold off

figure()
hold on
% PLOT C(X, TX)
subplot(2, 2, 1)
plot(iters, L1_hist, '-')
title('L1 COST')

% PLOT (F1 - F2)
subplot(2, 2, 2)
plot(iters, L2_hist, '-')
title('L2 COST')

% PLOT GLOBAL COST
subplot(2, 2, 3)
plot(iters, L_hist, '-')
title('L COST')

% PLOT LEARNING RATE/STEP SIZE ETA
subplot(2, 2, 4)
plot(iters, eta_hist, '-')
title('ETA')
hold off

% PLOT 25 MAP LOCATION HISTORIES
figure()
hold on
for i = 1:iter_num
%    if mod(i, floor(iter_num/25)) == 0
    T_map = T_hist(:,:,i);
    subplot(5, 4, i)
    hold on
    scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
    scatter3(T_map(:,1), T_map(:,2), T_map(:,3), 'filled', 'green')
    iterations = sprintf('ITERS: %d', i);
    title(iterations)
    hold off
%    if count < 25
%        count = count + 1;
%    end
%    end
end
hold off

% PLOT TRAJECTORY OF EACH POINT
figure()
hold on
for i = 1:iter_num
    T_map_i1 = T_hist(:,:,i);
    T_map_i2 = T_hist(:,:,i+1);
    for j = 1:length(x)
        plot3([T_map_i1(j,1) T_map_i2(j,1)], [T_map_i1(j,2) T_map_i2(j,2)], [T_map_i1(j,3) T_map_i2(j,3)], color = 'green')
    end
end

% PLOTTING FINAL OPTIMAL MAP
T_map = T_hist(:,:,iter_num+1);
%scatter3(x(:,1), x(:,2), x(:,3), 'filled', 'blue')
scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
scatter3(T_map(:,1), T_map(:,2), T_map(:,3), 'filled', 'green')
title('FINAL MAP')
hold off

% NEAREST NEIGHBOR SEARCH
%idk = knnsearch(T_map, y);
%new = T_map(idk,:);

% PLOTTING NEAREST TARGET POINTS TO OPTIMAL MAP RESULTS
%figure()
%hold on
%scatter3(new(:,1), new(:,2), new(:,3), 'filled', 'green')
%scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red');
%title('NEAREST NEIGHBORS RESULT')
%hold off

% COMPARING TARGET SET BEFORE AND AFTER NEAREST NEIGHBORS SEARCH
%figure()
%hold on
%subplot(1, 2, 1);
%scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
%title('INITIAL Y');

%subplot(1, 2, 2);
%scatter3(new(:,1), new(:,2), new(:,3), 'filled', 'green');
%title('FINAL');
%hold off


% END TIMER AND DISPLAY RUNTIME
plotting = toc;
disp(['ALGO RUNTIME: ' num2str(runtime) ' sec'])
disp(['PLOT RUNTIME: ' num2str(plotting) ' sec'])
disp(['TOTAL RUNTIME: ' num2str(runtime + plotting) ' sec'])


% BEGINNING OPTIMAL TRANSPORT ALGORITHM
% BANDWIDTH MATRIX SELECTION WITH SILVERMAN'S RULE OF THUMB
function H = bandwidth(x, n)
    [~, d] = size(x);                               
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        H = 0.9*min(std2(x), iqr(x)/1.34)*n^(-1/5);
    else      % FOR MULTIDIMENSIONAL CASE
        H = mean(std2(x))*(4/((d+2)*n))^(1/(d+4));
    end
end

% COST FUNCTION C (RETURNS CONSTANT)
function cost = C(x, Tx, weights)
    [n, d] = size(x);
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        cost = (norm(x - Tx)^2) / (2*n);
    else      % FOR MULTIDIMENSIONAL CASE
        cost = sum(weights.*norm(x - Tx, 'fro')^2) / (2*n);
    end
end

% WEIGHT FUNCTION USING MULTIDIMENSIONAL GAUSSIAN DENSITY
function weights = Weights(x1, x2, Hx)
    [n, d] = size(x1);
    func1 = zeros(d,n,n);
    for i = 1:d
        func1(i,:,:) = (x1(:,i)' - x2(:,i))./Hx;
    end
    f1 = sum(exp(-1/2.*sum(func1.^2, 1)), 3);
    const1 = 1/(n^2 * (Hx*sqrt(2*pi))^d);
    weights = const1.*(f1)';
end

% TEST FUNCTION DEFINING DISTANCE BETWEEN MAP AND TARGET (RETURNS CONSTANT)
function test = F(Tx1, y, Tx2, Hx, Hy, weights)
    [n, d] = size(Tx1);
    m = length(y);

    % INITIALIZE EMPTY MATRICES
    func1 = zeros(d,n,n);
    func2 = zeros(d,n,m);
    func3 = zeros(d,m,n);
    func4 = zeros(d,m,m);

    % FILL IN EACH MATRIX WITH THEIR RESPECTIVE VALUES AT EACH D
    for i = 1:d
        % PART F1 WHERE CENTER POINTS ARE IN Tx
        func1(i,:,:) = (Tx1(:,i)' - Tx2(:,i))./Hx;
        func2(i,:,:) = (y(:,i)' - Tx2(:,i))./Hy;

        % PART F2 WHERE CENTER POINTS ARE IN Y
        func3(i,:,:) = (Tx1(:,i)' - y(:,i))./Hx;
        func4(i,:,:) = (y(:,i)' - y(:,i))./Hy;
    end

    % USE TWO SUM FUNCTIONS IN PLACE OF DOUBLE FOR-LOOP
    f1 = sum(weights.*exp(-1/2.*sum(func1.^2, 1)), 'all');
    f2 = sum(weights.*exp(-1/2.*sum(func2.^2, 1)), 'all');
    f3 = sum(weights.*exp(-1/2.*sum(func3.^2, 1)), 'all');
    f4 = sum(weights.*exp(-1/2.*sum(func4.^2, 1)), 'all');

    weight_sum = sum(weights);
    
    % CONSTANTS IN FRONT OF SUM
    const1 = 1/(n^2 * (Hx*sqrt(2*pi))^d * weight_sum);
    const2 = 1/(m*n * (Hy*sqrt(2*pi))^d * weight_sum);
    const3 = 1/(n*m * (Hx*sqrt(2*pi))^d * weight_sum);
    const4 = 1/(m^2 * (Hy*sqrt(2*pi))^d * weight_sum);
    
    % FINAL RESULT MULTIPLYING CONSTANTS
    test = const1.*(f1)' - const2.*(f2)' - const3.*(f3)' + const4.*(f4)';
end

% GLOBAL COST FUNCTION L (RETURNS CONSTANT)
function l = L(x, y, Tx, Hx, Hy, lam, weights)
    l = C(x, Tx, weights) + lam*(F(Tx, y, Tx, Hx, Hy, weights));
end

% GRADIENT OF COST (RETURNS 1xD VECTOR)
function gradC = C_grad(x, Tx, weights)
    [n, d] = size(x);

    % INITIALIZE L GRADIENT TO ZEROS
    cgrad = zeros(n,d);

    % ADD VALUE TO L GRADIENT MATRIX AT EACH I
    for i = 1:n
        cgrad(i,:) = (Tx(i,:) - x(i,:)) ./ length(x(i,:));
    end
    gradC = weights .* cgrad;
end

% GRADIENT OF TEST FUNCTION TAKEN WITH RESPECT TO Tx CENTER
function gradF = F_grad(Tx1, y, Tx2, Hx, Hy, weights)
    [n, d] = size(Tx1);
    m = length(y);

    % INITIALIZE EMPTY MATRICES
    func1 = zeros(d,n,n);
    func2 = zeros(d,n,m);

    % FILL IN EACH MATRIX WITH THEIR RESPECTIVE VALUES AT EACH D
    for i = 1:d
        func1(i,:,:) = (Tx1(:,i)' - Tx2(:,i))./Hx;
        func2(i,:,:) = (y(:,i)' - Tx2(:,i))./Hy;
    end

    % USE TWO SUM FUNCTIONS IN PLACE OF DOUBLE FOR-LOOP
    f1 = sum(func1.*exp(-1/2.*sum(func1.^2, 1)), 3);
    f2 = sum(func2.*exp(-1/2.*sum(func2.^2, 1)), 3);

    weight_sum = sum(weights);

    % CONSTANTS IN FRONT OF SUM
    c1 = 1/(n^2 * (Hx*sqrt(2*pi))^d * Hx * weight_sum);
    c2 = 1/(m*n * (Hy*sqrt(2*pi))^d * Hy * weight_sum);
    
    % FINAL GRADIENT VALUE
    gradF = weights.*(c1.*(f1)' - c2.*(f2)');
end

% GRADIENT OF L (RETURNS NxD MATRIX)
function gradL = L_grad(x, y, Tx, Hx, Hy, lam, weights)
    gradL = C_grad(x, Tx, weights) + lam * F_grad(Tx, y, Tx, Hx, Hy, weights);
end

% ADAPTIVE LEARNING RATE ETA (RETURNS CONSTANT AND GRAD DESCENT RESULT)
function [eta, Tx_next] = adapt_learning(x, y, Tx_curr, Hx, Hy, lam, eta, weights)
    eta = eta * 2;                                                                 % INCREASE ETA FOR FASTER CONVERGENCE
    Tx_next = Tx_curr - (eta .* L_grad(x, y, Tx_curr, Hx, Hy, lam, weights));      % COMPUTE NEW MAP TX
    L_curr = L(x, y, Tx_curr, Hx, Hy, lam, weights);                               % COMPUTE COST BASED ON PAST MAP
    L_next = L(x, y, Tx_next, Hx, Hy, lam, weights);                               % COMPUTE COST BASED ON NEW MAP
    max_while = 5;
    while L_curr < L_next % NEXT COST L SHOULD NOT BE GREATER THAN THE CURRENT ONE
        if max_while < 0  % MAX LENGTH OF LOOP SHOULD NOT EXCEED 5 OR ETA WILL BECOME TOO SMALL
            break
        end
        eta = eta / 2;                                                             % SHRINK STEP SIZE
        Tx_next = Tx_curr - (eta .* L_grad(x, y, Tx_curr, Hx, Hy, lam, weights));  % COMPUTE NEW MAP TX WITH NEW ETA
        L_next = L(x, y, Tx_next, Hx, Hy, lam, weights);                           % COMPUTE COST BASED ON NEW MAP
        max_while = max_while - 1;                                                 % KEEP TRACK OF WHILE-LOOP LENGTH
    end
end

% LINEARLY INCREASING LAMBDA AND DECREASING BANDWIDTH (RETURNS DxD MATRIX)
function [Hz_new, lam_new] = linear_change(Hy, Hz, i, it, H_const, lam_init, lam_final)
    Hz_new = (H_const * Hz * (it - i) / it) + (Hy * i / it);
    lam_new = (lam_init * (it - i) / it) + (lam_final * i / it); 
end

% GRADIENT DESCENT
function [T_hist, L1_hist, L2_hist, L_hist, eta_hist] = grad_descent(x, y, eta, iter_num, H_const, lam_init, lam_final)

    % INITIALIZING EMPTY HISTORY FOR ALL PLOTS OVER TIME
    T_hist = x; % FIRST ENTRY IN MAP HISTORY SHOULD BE THE SOURCE DISTRIBUTION
    eta_hist = zeros(iter_num, 1);
    L1_hist = zeros(iter_num, 1);
    L2_hist = zeros(iter_num, 1);
    L_hist = zeros(iter_num, 1);

    % INITIAL VALUES
    z = [x; y];                         % COMBINED SET OF POINTS - FOR BANDWIDTH
    Hy = bandwidth(y, length(z));       % BANDWIDTH FOR Y
    Hz_init = bandwidth(z, length(z));  % BANDWIDTH FOR ALL POINTS
    Tx = x;                             % INITIAL MAP SHOULD BE THE ORIGINAL SET OF POINTS

    % LOOP OVER ITERATIONS
    for i = 1:iter_num
        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(i, 1) == 0
            fprintf("Iteration: %d\n", i)
        end

        % GETTING NEW BANDWIDTH (DECREASE TO HY) AND LAMBDA (INCREASE TO FINAL)
        [Hz, lam] = linear_change(Hy, Hz_init, i, iter_num, H_const, lam_init, lam_final);

        % WEIGHTS OF EACH PIXEL BASED ON POSITION IN DENSITY PLOT
        weights = Weights(Tx, Tx, Hz);

        % GET NEW MAP TX AND LEARNING RATE ETA AT EACH STEP
        [eta, Tx] = adapt_learning(x, y, Tx, Hz, Hz, lam, eta, weights);

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        T_hist(:,:,i+1) = Tx;
        eta_hist(i,:) = eta;
        L1_hist(i,:) = C(x, Tx, weights);
        L2_hist(i,:) = F(Tx, y, Tx, Hz, Hz, weights);
        L_hist(i,:) = L1_hist(i,:) + lam*(L2_hist(i,:));
    end
end