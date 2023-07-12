% MULTIDIMENSIONAL OPTIMAL TRANSPORT

% RANDOMLY GENERATED 2D POINTS
rng('default');

% STARTING PARAMETERS
eta = 0.1;          % INITIAL STEP SIZE
lam = 500000;          % REGULARIZATION PARAMETER (HIGHER = BETTER ALIGNMENT BUT MORE ITERATIONS)

%time_hist = zeros(40,1);
%iter_hist = zeros(40,1);
%L_final = zeros(20,1);

%for i = 1:20
%    disp(i)
    
% SYNTHETIC DATA IN THE SHAPE OF A GRID
a = linspace(0,4,5);
b = linspace(0,4,5);
[A, B] = meshgrid(a, b);

x_old = [A(:) B(:)];
y = normrnd(7, 0.5, [25,2]);

% PRECONDITIONING
x1 = (x_old).*std(y)./std(x_old);
x = x1 - mean(x1) + mean(y);


% START TIMER FOR ALGORITHM
tic

% RUNNING GRADIENT DESCENT
[T_hist, L1_hist, L2_hist, L_hist, eta_hist, iter, min_index] = grad_descent(x, y, eta, lam);
iters = 1:iter;
fprintf("\nTotal iters: %d\n", iter)
fprintf("Minimum achieved at iter: %d\n", min_index)
fprintf("Final cost: %d\n\n", L2_hist(min_index,:))

% MAP RUNTIME
runtime = toc;
%time_hist(i,:) = runtime;
%iter_hist(i,:) = iter;
%L_final(i,:) = L2_hist(iter);
%end

% PLOTTING INITIAL DISTRIBUTIONS
figure()
hold on
scatter(x_old(:,1), x_old(:,2), 'filled', 'blue')
scatter(y(:,1), y(:,2), 'filled', 'red')
title('INITIAL')
hold off

% PLOTTING DISTRIBUTIONS AFTER PRECONDITIONING
figure()
hold on
scatter(x(:,1), x(:,2), 'filled', 'blue')
scatter(y(:,1), y(:,2), 'filled', 'red')
title('PRECONDITIONED')
hold off

% START TIMER FOR PLOTTING
tic

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
count = 1;
for i = 1:min_index
    if mod(i, floor(min_index/25)) == 0
        T_map = T_hist(:,:,i);
        subplot(5, 5, count)
        hold on
        scatter(y(:,1), y(:,2), 'filled', 'red')
        scatter(T_map(:,1), T_map(:,2), 'filled', 'green')
        iterations = sprintf('ITERS: %d', i);
        title(iterations)
        hold off
        if count < 25
            count = count + 1;
        end
    end
end
hold off

%{
% PLOT TRAJECTORY OF EACH POINT
figure()
hold on
for i = 1:iter
    T_map_i1 = T_hist(:,:,i);
    T_map_i2 = T_hist(:,:,i+1);
    for j = 1:length(x)
        plot([T_map_i1(j,1) T_map_i2(j,1)], [T_map_i1(j,2) T_map_i2(j,2)], color = 'green')
    end
end
%}

% PLOTTING FINAL OPTIMAL MAP
figure()
hold on
T_map = T_hist(:,:,min_index);
%scatter(x(:,1), x(:,2), 'filled', 'blue')
scatter(y(:,1), y(:,2), 'filled', 'red')
scatter(T_map(:,1), T_map(:,2), 'filled', 'green')
title('FINAL MAP')
hold off

%{
% NEAREST NEIGHBOR SEARCH
idk = knnsearch(y, T_map);
new = y(idk,:);

% PLOTTING NEAREST TARGET POINTS TO OPTIMAL MAP RESULTS
figure()
hold on
scatter(T_map(:,1), T_map(:,2), 'filled', 'green');
scatter(new(:,1), new(:,2), 'filled', 'red');
title('NEAREST NEIGHBORS RESULT')
hold off

% COMPARING TARGET SET BEFORE AND AFTER NEAREST NEIGHBORS SEARCH
figure()
hold on
subplot(1, 2, 1);
scatter(y(:,1), y(:,2), 'filled', 'red');
title('INITIAL Y');

subplot(1, 2, 2);
scatter(new(:,1), new(:,2), 'filled', 'red');
title('FINAL Y');
hold off
%}

% END TIMER AND DISPLAY RUNTIME
plotting = toc;
disp(['ALGO RUNTIME: ' num2str(runtime) ' sec'])
disp(['PLOT RUNTIME: ' num2str(plotting) ' sec'])
disp(['TOTAL RUNTIME: ' num2str(runtime + plotting) ' sec'])
%}

% BEGINNING OPTIMAL TRANSPORT ALGORITHM
% BANDWIDTH MATRIX SELECTION WITH SILVERMAN'S RULE OF THUMB
function H = bandwidth(x, n)
    [~, d] = size(x);                               
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        H = 0.9*min(std(x), iqr(x)/1.34)*n^(-1/5);
    else      % FOR MULTIDIMENSIONAL CASE
        H = mean(std(x))*(4/((d+2)*n))^(1/(d+4));
    end
end

% COST FUNCTION C (RETURNS CONSTANT)
function cost = C(x, Tx)
    [n, d] = size(x);
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        cost = (norm(x - Tx)^2) / (2*n);
    else      % FOR MULTIDIMENSIONAL CASE
        cost = (norm(x - Tx, 'fro')^2) / (2*n);
    end
end

% TEST FUNCTION DEFINING DISTANCE BETWEEN MAP AND TARGET (RETURNS CONSTANT)
function test = F(Tx1, y, Tx2, Hx, Hy)
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
    f1 = sum(exp(-1/2.*sum(func1.^2, 1)), 'all');
    f2 = sum(exp(-1/2.*sum(func2.^2, 1)), 'all');
    f3 = sum(exp(-1/2.*sum(func3.^2, 1)), 'all');
    f4 = sum(exp(-1/2.*sum(func4.^2, 1)), 'all');
    
    % CONSTANTS IN FRONT OF SUM
    const1 = 1/(n^2 * (Hx*sqrt(2*pi))^d);
    const2 = 1/(m*n * (Hy*sqrt(2*pi))^d);
    const3 = 1/(n*m * (Hx*sqrt(2*pi))^d);
    const4 = 1/(m^2 * (Hy*sqrt(2*pi))^d);
    
    % FINAL RESULT MULTIPLYING CONSTANTS
    test = const1.*(f1)' - const2.*(f2)' - const3.*(f3)' + const4.*(f4)';
end

% GLOBAL COST FUNCTION L (RETURNS CONSTANT)
function l = L(x, y, Tx, Hx, Hy, lam)
    l = C(x, Tx) + lam*(F(Tx, y, Tx, Hx, Hy));
end

% GRADIENT OF COST (RETURNS 1xD VECTOR)
function gradC = C_grad(x, Tx)
    [n, d] = size(x);

    % INITIALIZE L GRADIENT TO ZEROS
    gradC = zeros(n,d);

    % ADD VALUE TO L GRADIENT MATRIX AT EACH I
    for i = 1:n
        gradC(i,:) = (Tx(i,:) - x(i,:)) ./ length(x(i,:));
    end
end

% GRADIENT OF TEST FUNCTION TAKEN WITH RESPECT TO Tx CENTER
function gradF = F_grad(Tx1, y, Tx2, Hx, Hy)
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

    % CONSTANTS IN FRONT OF SUM
    c1 = 1/(n^2 * (Hx*sqrt(2*pi))^d * Hx);
    c2 = 1/(m*n * (Hy*sqrt(2*pi))^d * Hy);
    
    % FINAL GRADIENT VALUE
    gradF = c1.*(f1)' - c2.*(f2)';
end

% GRADIENT OF L (RETURNS NxD MATRIX)
function gradL = L_grad(x, y, Tx, Hx, Hy, lam)
    gradL = C_grad(x, Tx) + lam * F_grad(Tx, y, Tx, Hx, Hy);
end

% ADAPTIVE LEARNING RATE ETA (RETURNS CONSTANT AND GRAD DESCENT RESULT)
function [eta, Tx_next] = adapt_learning(x, y, Tx_curr, Hx, Hy, lam, eta)
    eta = eta * 2;                                                        % INCREASE ETA FOR FASTER CONVERGENCE
    l_grad = L_grad(x, y, Tx_curr, Hx, Hy, lam);
    Tx_next = Tx_curr - (eta .* l_grad);                                  % COMPUTE NEW MAP TX
    L_curr = L(x, y, Tx_curr, Hx, Hy, lam);                               % COMPUTE COST BASED ON PAST MAP
    L_next = L(x, y, Tx_next, Hx, Hy, lam);                               % COMPUTE COST BASED ON NEW MAP
    while L_curr < L_next % NEXT COST L SHOULD NOT BE GREATER THAN THE CURRENT ONE
        eta = eta / 2;                                                    % SHRINK STEP SIZE
        Tx_next = Tx_curr - (eta .* l_grad);                              % COMPUTE NEW MAP TX WITH NEW ETA
        L_next = L(x, y, Tx_next, Hx, Hy, lam);                           % COMPUTE COST BASED ON NEW MAP
    end
end

% GRADIENT DESCENT
function [T_hist, L1_hist, L2_hist, L_hist, eta_hist, iter, min_index] = grad_descent(x, y, eta, lam)
    % INITIAL VALUES
    Tx = x;                             % INITIAL MAP SHOULD BE THE ORIGINAL SET OF POINTS
    z = [Tx; y];                         % COMBINED SET OF POINTS - FOR BANDWIDTH
    Hy = bandwidth(y, length(x));       % BANDWIDTH FOR Y
    Hz = bandwidth(z, length(x));  % BANDWIDTH FOR ALL POINTS

    % INITIALIZING EMPTY HISTORY FOR ALL PLOTS OVER TIME
    T_hist = Tx; % FIRST ENTRY IN MAP HISTORY SHOULD BE THE SOURCE DISTRIBUTION
    eta_hist = eta;
    L1_hist = C(x, Tx);
    L2_hist = F(Tx, y, Tx, Hz, Hy);
    L_hist = L1_hist + lam * F(Tx, y, Tx, Hz, Hy);

    criteria = 1e8; % ARBITRARY LARGE NUMBER TO START
    minimum = 1e8;
    iter = 1;
    min_index = 1;

    % CONTINUE UNTIL REACHING STOPPING CRITERIA
    while criteria > 0
        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(iter, 100) == 0
            fprintf("Iteration: %d\n", iter)
        end

        if iter == 1500
            break
        end
        iter = iter+1;

        % GETTING NEW BANDWIDTH (DECREASE TO HY) AND LAMBDA (INCREASE TO FINAL)
        z = [Tx; y];
        Hz = bandwidth(z, length(x));
        Hz = (Hz + Hy) / 2;

        % GET NEW MAP TX AND LEARNING RATE ETA AT EACH STEP
        [eta, Tx] = adapt_learning(x, y, Tx, Hz, Hz, lam, eta);

        criteria = F(Tx, y, Tx, Hz, Hy);

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        T_hist(:,:,iter) = Tx;
        eta_hist(iter,:) = eta;
        L1_hist(iter,:) = C(x, Tx);
        L2_hist(iter,:) = criteria;
        L_hist(iter,:) = L1_hist(iter,:) + lam*(L2_hist(iter,:));

        if abs(criteria) < abs(minimum)
            minimum = criteria;
            min_index = iter;
        else
            break
        end
    end
end