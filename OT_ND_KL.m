% MULTIDIMENSIONAL OPTIMAL TRANSPORT

% RANDOMLY GENERATED 2D POINTS
rng('default');

% STARTING PARAMETERS
eta = 0.1;          % INITIAL STEP SIZE
lam = 5e5;          % TRANSPORT REGULARIZATION

%time_hist = zeros(40,1);
%iter_hist = zeros(40,1);
%L_final = zeros(20,1);
%for i = 1:20
%    disp(i)
    
% SLOPE DATA SETS
%x_old = table2array(readtable('revised data edit.xlsx', Sheet='slope'));
%x = table2array(readtable('revised data edit.xlsx', Sheet='slope'));
%y = table2array(readtable('revised data edit.xlsx', Sheet='slope (2)'));

% PYRAMID DATA SETS
y = table2array(readtable('revised data 3.xlsx', Sheet='every'));
x = table2array(readtable('revised data set 1.xlsx', Sheet='every (3)'));

% PYRAMID DATA WITH SHIFTED SOURCE
%x = table2array(readtable('revised data 3.xlsx', Sheet='every (3)'));
%y = table2array(readtable('revised data set 1.xlsx', Sheet='every'));

% 2D DATA POINTS
a = linspace(0,4,20);
b = linspace(0,4,20);
[A, B] = meshgrid(a, b);
%x = [A(:) B(:)];
mu_x = [0 9];
sigma_x = [0.25 0; 0 1];
x = mvnrnd(mu_x, sigma_x, 10);
x_original = x;

mu_y1 = [7 7];
sigma_y1 = [1 0; 0 0.25];
%y = mvnrnd(mu_y1, sigma_y1, 200);
mu_y2 = [7 11];
sigma_y2 = [0.1 0; 0 0.1];
sigma_y3 = [10 0; 0 10];

y1 = mvnrnd(mu_y1, sigma_y1, 10);
y2 = mvnrnd(mu_y2, sigma_y2, 1);
y3 = mvnrnd(mu_y1, sigma_y3, 3);
y = [y1; y2; y3];

% TEST
y = x(1:6, 1:2);

% MATCH THE DIMENSIONS
[n, d_x] = size(x);
[m, d_y] = size(y);
dim_diff = abs(d_x - d_y);

if d_x > d_y
    newrow_y = zeros(m, dim_diff);
    y = [y, newrow_y];
elseif d_x < d_y
    newrow_x = zeros(n, dim_diff);
    x = [x, newrow_x];
end

% PLOTTING INITIAL DISTRIBUTIONS
figure()
hold on
if d_x == 2 && d_y == 2
    scatter(y(:,1), y(:,2), 'filled', 'red')
    scatter(x(:,1), x(:,2), 'filled', 'blue')
elseif d_x == 3 || d_y == 3
    scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
    scatter3(x(:,1), x(:,2), x(:,3), 'filled', 'blue')
end
title('INITIAL DISTRIBUTIONS')
legend('SOURCE', 'TARGET')
hold off

% PRECONDITIONING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
s_x = sqrt(sum(abs(x-median(x)))/length(x)); % IMMUNE TO LARGE CLUSTERS OF OUTLIERS
s_y = sqrt(sum(abs(y-median(y)))/length(y)); % ROBUST DISPERSION MEASURE USING L1 NORM

%x1 = x.*std(y)./std(x);    % OG RESCALING BASED ON STD - BETTER FOR SMALL OUTLIERS
x1 = x.*s_y./s_x;          % RESCALING X1 BASED ON ROBUST MEASURE
%x1 = x.*s_y./std(x);           % USE STD OF X TO REACH MORE POINTS IN X

% SHIFTING TO MEAN OR MEDIAN OF TARGET
%x = x1 - mean(x1) + mean(y);
x = x1 - median(x1) + median(y);

% ADDING ANOTHER DIMENSION
if d_x > d_y
    y = [y(:,1), y(:,2), ones(m, dim_diff)*mean(y(:,3))];
elseif d_x < d_y
    x = [x(:,1), x(:,2), ones(n, dim_diff)*mean(y(:,3))];
end


% START TIMER FOR ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic

% RUNNING GRADIENT DESCENT
[T_hist, L1_hist, L2_hist, L_hist, eta_hist, H_hist, iter, min_index] = grad_descent(x, y, eta, lam);
iters = 1:iter;
fprintf("\nTotal iters: %d\n", iter)
fprintf("Minimum achieved at iter: %d\n", min_index)
fprintf("Final cost: %d\n\n", L_hist(min_index,:))

% MAP RUNTIME
runtime = toc;

% PLOTTING INITIAL DISTRIBUTIONS
figure()
hold on
if d_x == 2 && d_y == 2
    scatter(y(:,1), y(:,2), 'filled', 'red')
    scatter(x(:,1), x(:,2), 'filled', 'blue')
elseif d_x == 3 || d_y == 3
    scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
    scatter3(x(:,1), x(:,2), x(:,3), 'filled', 'blue')
end
title('PRECONDITIONED DISTRIBUTIONS')
legend('SOURCE', 'TARGET')
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

% PLOTTING FINAL OPTIMAL MAP
figure()
hold on

for i = 1:iter-1
    T_map_i1 = T_hist(:,:,i);
    T_map_i2 = T_hist(:,:,i+1);
    for j = 1:length(x)
        plot([T_map_i1(j,1) T_map_i2(j,1)], [T_map_i1(j,2) T_map_i2(j,2)], color = 'green')
    end
end

T_map = T_hist(:,:,min_index);
if d_x == 2 && d_y == 2
    p_y = scatter(y(:,1), y(:,2), 'filled', 'red');
    p_t = scatter(T_map(:,1), T_map(:,2), 'filled', 'green');
elseif d_x == 3 || d_y == 3
    p_y = scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red');
    p_t = scatter3(T_map(:,1), T_map(:,2),  T_map(:,3), 'filled', 'green');
end
title('FINAL MAP')
legend([p_y, p_t], {'TARGET', 'MAP'})
hold off

% END TIMER AND DISPLAY RUNTIME
plotting = toc;
disp(['ALGO RUNTIME: ' num2str(runtime) ' sec'])
disp(['PLOT RUNTIME: ' num2str(plotting) ' sec'])
disp(['TOTAL RUNTIME: ' num2str(runtime + plotting) ' sec'])


% BEGINNING OPTIMAL TRANSPORT ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BANDWIDTH MATRIX SELECTION WITH SILVERMAN'S RULE OF THUMB
function H = bandwidth(x, n, std_switch)
    [~, d] = size(x);
    if std_switch == false
        if d == 1 % FOR ONE-DIMENSIONAL CASE
            H = 0.9*min(std(x), iqr(x)/1.34)*n^(-1/5);
        else
            H = std(x)*(4/((d+2)*n))^(1/(d+4));
        end
    else
        s_x = sqrt(sum(abs(x-median(x)))/length(x)); % IMMUNE TO LARGE CLUSTERS OF OUTLIERS
        if d == 1 % FOR ONE-DIMENSIONAL CASE
            H = 0.9*min(s_x, iqr(x)/1.34)*n^(-1/5);
        else
            H = s_x*(4/((d+2)*n))^(1/(d+4));
        end
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

% TEST FUNCTION DEFINING KL DIVERGENCE MAP AND TARGET (RETURNS CONSTANT)
function test = KL(Tx1, y, Tx2, Hx, Hy)
    [n, d] = size(Tx1);
    m = length(y);
    epsilon = 1e-10; % TO MAKE SURE WE DON'T DIVIDE BY ZERO
    
    % INITIALIZE EMPTY MATRICES
    p = zeros(d,n,n);
    q = zeros(d,n,m);

    % FILL IN EACH MATRIX WITH THEIR RESPECTIVE VALUES AT EACH D
    for i = 1:d
        % DISTRIBUTION P(x) WITH CENTER POINTS IN Tx
        p(i,:,:) = (Tx1(:,i)' - Tx2(:,i))./Hx(:,i);
        
        % DISTRIBUTION Q(x) WITH CENTER POINTS IN Tx
        q(i,:,:) = (y(:,i)' - Tx2(:,i))./Hy(:,i);
    end

    % USE TWO SUM FUNCTIONS IN PLACE OF DOUBLE FOR-LOOP
    p1 = sum(exp(-1/2.*sum(p.^2, 1)), 3);
    q1 = sum(exp(-1/2.*sum(q.^2, 1)), 3);

    % CONSTANTS TO NORMALIZE AND FINISH COMPUTATION
    const1 = 1/(n^2 * (mean(Hx)*sqrt(2*pi)).^d + epsilon);
    const2 = 1/(m*n * (mean(Hy)*sqrt(2*pi)).^d + epsilon);
    
    % FINAL DISTRIBUTIONS
    p_dist = const1 .* p1;
    q_dist = const2 .* q1;

    % ENTROPY DEFINING POINT-BY-POINT DIFFERENCE
    entropy = log(p_dist) - log(q_dist);

    % FINAL RESULT SUMMING TOTAL ENTROPY AND NORMALIZING
    test = sum(entropy)/n;
end

% GLOBAL COST FUNCTION L (RETURNS CONSTANT)
function l = L(x, y, Tx, Hx, Hy, lam)
    l = C(x, Tx) + lam*(KL(Tx, y, Tx, Hx, Hy));
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
function gradF = KL_grad(Tx1, y, Tx2, Hx, Hy)
    [n, d] = size(Tx1);
    m = length(y);
    epsilon = 1e-10;

    % INITIALIZE EMPTY MATRICES
    p = zeros(d,n,n);
    q = zeros(d,n,m);

    % FILL IN EACH MATRIX WITH THEIR RESPECTIVE VALUES AT EACH D
    for i = 1:d
        % DISTRIBUTION P(x) WITH CENTER POINTS IN Tx
        p(i,:,:) = (Tx1(:,i)' - Tx2(:,i))./Hx(:,i);

        % DISTRIBUTION Q(x) WITH CENTER POINTS IN Tx
        q(i,:,:) = (y(:,i)' - Tx2(:,i))./Hy(:,i);
    end

    % SUMMATION TO CALCULATE ORIGINAL DISTRIBUTION
    p1 = sum(exp(-1/2.*sum(p.^2, 1)), 'all') + epsilon;
    q1 = sum(exp(-1/2.*sum(q.^2, 1)), 'all') + epsilon;

    % USE TWO SUM FUNCTIONS IN PLACE OF DOUBLE FOR-LOOP FOR GRADIENT
    grad_p1 = sum(p.*exp(-1/2.*sum(p.^2, 1)), 3)';
    grad_q1 = sum(q.*exp(-1/2.*sum(q.^2, 1)), 3)';

    % CONSTANTS IN FRONT OF ORIGINAL DISTRIBUTION SUM
    const1 = 1/(n^2 * (mean(Hx)*sqrt(2*pi)).^d + epsilon);
    const2 = 1/(m*n * (mean(Hy)*sqrt(2*pi)).^d + epsilon);

    % CONSTANTS IN FRONT OF GRADIENT SUM
    grad_p_const = 1/(n^2 * (mean(Hx)*sqrt(2*pi)).^d * mean(Hx) + epsilon);
    grad_q_const = 1/(m*n * (mean(Hy)*sqrt(2*pi)).^d * mean(Hy) + epsilon);

    % FINAL GRADIENT VALUE
    gradF = (grad_p_const.*grad_p1)./(n*const1.*p1) - (grad_q_const.*grad_q1)./(n*const2.*q1);
    %gradF = grad_p1./(n*p1) - grad_q1./(n*q1);
end

% GRADIENT OF L (RETURNS NxD MATRIX)
function gradL = L_grad(x, y, Tx, Hx, Hy, lam)
    gradL = C_grad(x, Tx) + lam * KL_grad(Tx, y, Tx, Hx, Hy);
end

% ADAPTIVE LEARNING RATE ETA (RETURNS CONSTANT AND GRAD DESCENT RESULT)
function [eta, Tx_next] = adapt_learning(x, y, Tx_curr, Hx, Hy, lam, eta)
    eta = eta * 2;                                                        % INCREASE ETA FOR FASTER CONVERGENCE
    l_grad = L_grad(x, y, Tx_curr, Hx, Hy, lam);
    Tx_next = Tx_curr - (eta .* l_grad);                                  % COMPUTE NEW MAP TX
    L_curr = L(x, y, Tx_curr, Hx, Hy, lam);                               % COMPUTE COST BASED ON PAST MAP
    L_next = L(x, y, Tx_next, Hx, Hy, lam);                               % COMPUTE COST BASED ON NEW MAP
    while L_curr < L_next                     
        eta = eta / 2;                                                    % SHRINK STEP SIZE
        Tx_next = Tx_curr - (eta .* l_grad);                              % COMPUTE NEW MAP TX WITH NEW ETA
        L_next = L(x, y, Tx_next, Hx, Hy, lam);                           % COMPUTE COST BASED ON NEW MAP
    end
end

% GRADIENT DESCENT
function [T_hist, L1_hist, L2_hist, L_hist, eta_hist, H_hist, iter, min_index] = grad_descent(x, y, eta, lam)
    % INITIAL VALUES
    Tx = x;                                  % INITIAL MAP SHOULD BE THE ORIGINAL SET OF POINTS
    z = [Tx; y];                             % COMBINED SET OF POINTS - FOR BANDWIDTH
    Hy = bandwidth(y, length(x), true);      % BANDWIDTH FOR Y - USE ROBUST DEVIATION MEASURE
    Hz = bandwidth(z, length(x), true);     % BANDWIDTH FOR ALL POINTS - USE STD

    % INITIALIZING EMPTY HISTORY FOR ALL PLOTS OVER TIME
    T_hist = Tx; % FIRST ENTRY IN MAP HISTORY SHOULD BE THE SOURCE DISTRIBUTION
    H_hist = Hz;
    eta_hist = eta;
    L1_hist = C(x, Tx);
    L2_hist = KL(Tx, y, Tx, Hz, Hy);
    L_hist = L1_hist + lam * KL(Tx, y, Tx, Hz, Hy);

    criteria = 1e10; % ARBITRARY LARGE NUMBER TO START
    minimum = 1e10;
    iter = 1;
    min_index = 1;

    % CONTINUE UNTIL REACHING STOPPING CRITERIA
    while criteria > 0
        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(iter, 100) == 0
            fprintf("Iteration: %d\n", iter)
        end

        if iter == 100
            break
        end
        iter = iter+1;

        % GETTING NEW BANDWIDTH (CONVERGE TO HY)
        z = [Tx; y];
        Hz = bandwidth(z, length(x), true);
        Hz = (Hz + Hy) / 2;

        % GET NEW MAP TX AND LEARNING RATE ETA AT EACH STEP
        [eta, Tx] = adapt_learning(x, y, Tx, Hz, Hz, lam, eta);

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        T_hist(:,:,iter) = Tx;
        H_hist(:,:,iter) = Hz;
        eta_hist(iter,:) = eta;
        L1_hist(iter,:) = C(x, Tx);
        L2_hist(iter,:) = KL(Tx, y, Tx, Hz, Hy);
        L_hist(iter,:) = L1_hist(iter,:) + lam*(L2_hist(iter,:));

        % DEFINING CRITERIA TO CONTINUE OR STOP ITERATION
        criteria = L_hist(iter,:);
        if abs(criteria) < abs(minimum)
            minimum = criteria;
            min_index = iter;
        else
            %break
        end
    end
end