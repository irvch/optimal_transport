% MULTIDIMENSIONAL OPTIMAL TRANSPORT

% RANDOMLY GENERATED 2D POINTS
rng('default');

% STARTING PARAMETERS
eta = 0.1;          % INITIAL STEP SIZE
lam = 5e5;          % REGULARIZATION PARAMETER
mu = 1e5;           % PARTIAL-NESS REGULARLIZATION

% 2D DATA POINTS
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
y = x;
x = x(1:6, 1:2);

% INITIAL WEIGHTS
delta = zeros(length(x), 1);
epsilon = zeros(length(y), 1);
p_init = exp(delta)./sum(exp(delta));
q_init = exp(epsilon)./sum(exp(epsilon));

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
    scatter(y(:,1), y(:,2), q_init*500, 'filled', 'red')
    scatter(x(:,1), x(:,2), p_init*500, 'filled', 'blue')
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

x1 = x.*s_y./s_x;          % RESCALING X1 BASED ON ROBUST MEASURE

% SHIFTING TO MEAN OR MEDIAN OF TARGET
x = x1 - median(x1) + median(y);

% START TIMER FOR ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic

% RUNNING GRADIENT DESCENT
[T_hist, L1_hist, L2_hist, L_hist, eta_hist, H_hist, iter, min_index, p_final, q_final] = grad_descent(x, y, eta, lam, mu, delta, epsilon);
iters = 1:iter;
fprintf("\nTotal iters: %d\n", iter)
fprintf("Minimum achieved at iter: %d\n", min_index)
fprintf("Final cost: %d\n\n", L_hist(min_index,:))
disp(p_final)
disp(q_final)

% MAP RUNTIME
runtime = toc;

% PLOTTING INITIAL DISTRIBUTIONS
figure()
hold on
if d_x == 2 && d_y == 2
    scatter(y(:,1), y(:,2), q_init*500, 'filled', 'red')
    scatter(x(:,1), x(:,2), p_init*500, 'filled', 'blue')
elseif d_x == 3 || d_y == 3
    scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red')
    scatter3(x(:,1), x(:,2), x(:,3), 'filled', 'blue')
end
title('PRECONDITIONED DISTRIBUTIONS')
legend('TARGET','SOURCE')
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
    p_y = scatter(y(:,1), y(:,2), q_final*500, 'filled', 'red');
    p_t = scatter(T_map(:,1), T_map(:,2), p_final*500, 'filled', 'green');
elseif d_x == 3 || d_y == 3
    p_y = scatter3(y(:,1), y(:,2), y(:,3), 'filled', 'red');
    p_t = scatter3(T_map(:,1), T_map(:,2),  T_map(:,3),  'filled', 'green');
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
function cost = C(x, Tx, p_hat)
    [n, d] = size(x);
    if d == 1 % FOR ONE-DIMENSIONAL CASE
        cost = (norm(x - Tx)^2) / (2*n);
    else      % FOR MULTIDIMENSIONAL CASE
        cost = sum(p_hat.*(x-Tx).^2, 'all') / (2*n);
    end
end




% TEST FUNCTION DEFINING KL DIVERGENCE MAP AND TARGET (RETURNS CONSTANT)
function test = KL(Tx1, y, Tx2, Hx, Hy, delta, epsilon)
    [n, d] = size(Tx1);
    m = length(y);
    bias = 1e-10; % TO MAKE SURE WE DON'T DIVIDE BY ZERO
    
    % INITIALIZE EMPTY MATRICES
    p = zeros(d,n,n);
    q = zeros(d,n,m);

    % FILL IN EACH MATRIX WITH THEIR RESPECTIVE VALUES AT EACH D
    for i = 1:d
        % DISTRIBUTION P(x) WITH CENTER POINTS IN Tx
        p(i,:,:) = (Tx1(:,i)' - Tx2(:,i))./Hx(:,i);
        
        % DISTRIBUTION Q(x) WITH CENTER POINTS IN Tx
        q(i,:,:) = -1*(y(:,i)' - Tx2(:,i))./Hy(:,i);
    end

    p_hat = exp(delta)/sum(exp(delta));
    q_hat = exp(epsilon)/sum(exp(epsilon));

    % MATRIX MULTIPLICATION IN PLACE OF DOUBLE FOR-LOOP
    p1 = sum(reshape(exp(-1/2.*sum(p.^2, 1)), [], length(p_hat))' .* p_hat, 1);
    q1 = sum(reshape(exp(-1/2.*sum(q.^2, 1)), [], length(q_hat))' .* q_hat, 1);

    %disp(reshape(exp(-1/2.*sum(p.^2, 1)), [], length(p_hat)))
    %disp(sum(reshape(exp(-1/2.*sum(p.^2, 1)), [], length(p_hat)), 2)');
    %disp(sum(reshape(exp(-1/2.*sum(p.^2, 1)), [], length(p_hat))' .* p_hat, 1))


    % CONSTANTS TO NORMALIZE AND FINISH COMPUTATION
    const1 = 1/(n^2 * (mean(Hx)*sqrt(2*pi)).^d + bias);
    const2 = 1/(m*n * (mean(Hy)*sqrt(2*pi)).^d + bias);
    
    % FINAL DISTRIBUTIONS
    p_dist = const1 .* p1;
    q_dist = const2 .* q1;

    % ENTROPY DEFINING POINT-BY-POINT DIFFERENCE
    entropy = log(p_dist) - log(q_dist);

    % FINAL RESULT SUMMING TOTAL ENTROPY AND NORMALIZING
    test = sum(entropy .* p_hat')/n;
end




% GLOBAL COST FUNCTION L (RETURNS CONSTANT)
function l = L(x, y, Tx, Hx, Hy, lam, mu, delta, epsilon)
    p_hat = exp(delta)/sum(exp(delta));
    q_hat = exp(epsilon)/sum(exp(epsilon));

    % IN PLACE OF MAX FUNCTION SINCE GRADIENT DNE
    big_num = 1e2;
    reg1 = sum(p_hat.^big_num).^(1/big_num);
    reg2 = sum(q_hat.^big_num).^(1/big_num);
    
    % FINAL RESULT
    l = C(x, Tx, delta) + lam*(KL(Tx, y, Tx, Hx, Hy, delta, epsilon)) + mu*(reg1+reg2);
end



% GRADIENT OF COST (RETURNS 1xD VECTOR)
function [gradC_Tx, gradC_d, gradC_e] = C_grad(x, Tx, delta)
    p_hat = exp(delta)/sum(exp(delta));
    gradC_Tx = p_hat.*((x-Tx)) / length(x);
    gradC_d = sum(p_hat.*(2*(x-Tx) + (x-Tx).^2) - p_hat.^2 .* (x-Tx).^2, 2) / 2*length(x);
    gradC_e = 0;
end



% GRADIENT OF TEST FUNCTION TAKEN WITH RESPECT TO Tx CENTER
function [gradF_Tx, gradF_d, gradF_e] = KL_grad(Tx1, y, Tx2, Hx, Hy, delta, epsilon)
    [n, d] = size(Tx1);
    m = length(y);
    bias = 1e-8; % TO MAKE SURE WE DON'T DIVIDE BY ZERO
    
    % INITIALIZE EMPTY MATRICES
    p = zeros(d,n,n);
    q = zeros(d,n,m);
    p_hat = exp(delta)/sum(exp(delta));
    q_hat = exp(epsilon)/sum(exp(epsilon));

    % FILL IN EACH MATRIX WITH THEIR RESPECTIVE VALUES AT EACH D
    for i = 1:d
        % DISTRIBUTION P(x) WITH CENTER POINTS IN Tx
        p(i,:,:) = (Tx1(:,i)' - Tx2(:,i))./Hx(:,i);
        
        % DISTRIBUTION Q(x) WITH CENTER POINTS IN Tx
        q(i,:,:) = -1*(y(:,i)' - Tx2(:,i))./Hy(:,i);                % WHY -1 IN FRONT?
    end

    % MATRIX MULTIPLICATION IN PLACE OF DOUBLE FOR-LOOP
    p1 = sum(reshape(exp(-1/2.*sum(p.^2, 1)), [], length(p_hat))' .* p_hat, 1);
    q1 = sum(reshape(exp(-1/2.*sum(q.^2, 1)), [], length(q_hat))' .* q_hat, 1);

    % CONSTANTS IN FRONT OF ORIGINAL DISTRIBUTION SUM
    const1 = 1/(n^2 * (mean(Hx)*sqrt(2*pi)).^d + bias);
    const2 = 1/(m*n * (mean(Hy)*sqrt(2*pi)).^d + bias);

    % ORIGINAL DISTRIBUTIONS
    %p_dist = const1 .* p1;
    %q_dist = const2 .* q1;
    
    % GRADIENT 
    % USE TWO SUM FUNCTIONS IN PLACE OF DOUBLE FOR-LOOP FOR GRADIENT
    grad_p1a = p.*exp(-1/2.*sum(p.^2, 1)); 
    grad_q1a = q.*exp(-1/2.*sum(q.^2, 1));

    for j=1:n
        grad_p1a(:,:,j) = grad_p1a(:,:,j) .* p_hat(j,:);
    end
    for k=1:m
        grad_q1a(:,:,k) = grad_q1a(:,:,k) .* q_hat(k,:);
    end

    grad_p1 = sum(grad_p1a, 3);
    grad_q1 = sum(grad_q1a, 3);
    
    % CONSTANTS IN FRONT OF GRADIENT SUM
    %grad_p_const = 1/(n^2 * (mean(Hx)*sqrt(2*pi)).^d * mean(Hx) + bias);
    %grad_q_const = 1/(m*n * (mean(Hy)*sqrt(2*pi)).^d * mean(Hy) + bias);

    % FINAL GRADIENT VALUES
    gradF_Tx = p_hat .* ((grad_p1./p1 - grad_q1./q1)/n)';
    gradF_d = sum(p_hat - p_hat.^2).*(log((const1*p1)./(const2*q1)) + 1)';
    gradF_e = sum(p_hat .* (1 - q_hat)', 1)';
    %gradF_d = zeros(n, 1);
    %gradF_e = zeros(m, 1);
end




% GRADIENT OF L WITH RESPECT TO TX, delta, and epsilon (RETURNS NxD MATRICES)
function [gradL_Tx, gradL_d, gradL_e] = L_grad(x, y, Tx, Hx, Hy, lam, mu, delta, epsilon)
    [gradC_Tx, gradC_d, gradC_e] = C_grad(x, Tx, delta);
    [gradF_Tx, gradF_d, gradF_e] = KL_grad(Tx, y, Tx, Hx, Hy, delta, epsilon);
    big_num = 1e2;

    p_hat = exp(delta)/sum(exp(delta));
    q_hat = exp(epsilon)/sum(exp(epsilon));

    % GRAD WITH RESPECT TO Tx
    gradL_Tx = gradC_Tx + lam*gradF_Tx;

    % GRAD WITH RESPECT TO p
    grad_reg1 = (sum(p_hat.^big_num).^(1/big_num - 1)) * p_hat.^(big_num-1);
    gradL_d = gradC_d + lam*gradF_d + mu*grad_reg1;
    
    % GRAD WITH RESPECT TO q
    grad_reg2 = (sum(q_hat.^big_num).^(1/big_num - 1)) * q_hat.^(big_num-1);
    gradL_e = gradC_e + lam*gradF_e + mu*grad_reg2;
end




% ADAPTIVE LEARNING RATE ETA (RETURNS CONSTANT AND GRAD DESCENT RESULT)
function [eta, Tx_next, d_next, e_next] = adapt_learning(x, y, Tx_curr, Hx, Hy, lam, mu, eta, d_curr, e_curr)
    [n, d] = size(Tx_curr);
    [m, ~] = size(y);

    % INCREASE ETA FOR FASTER CONVERGENCE
    eta = eta * 2;
    [Tx_grad, d_grad, e_grad] = L_grad(x, y, Tx_curr, Hx, Hy, lam, mu, d_curr, e_curr);

    % ITERATE THROUGH DIFFERENT COMBOS OF UPDATES TO FIND THE BEST ONE    
    combos = [0 0 0; 1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; 1 1 1];
    num_combos = size(combos, 1);

    L_curr = L(x, y, Tx_curr, Hx, Hy, lam, mu, d_curr, e_curr);
    next_L = zeros(num_combos, 1);
    next_T = zeros(num_combos, n, d);
    next_d = zeros(num_combos, n, 1);
    next_e = zeros(num_combos, m, 1);

    for i=1:num_combos
        if combos(i,1) == 1
            next_T(i,:,:) = Tx_curr - (eta * Tx_grad);
        else
            next_T(i,:,:) = Tx_curr;
        end
        if combos(i,2) == 1
            next_d(i,:,:) = d_curr - (eta * d_grad);
        else
            next_d(i,:,:) = d_curr;
        end
        if combos(i,3) == 1
            next_e(i,:,:) = e_curr - (eta * e_grad);
        else
            next_e(i,:,:) = e_curr;
        end
        T_test = reshape(next_T(i,:,:), n, d);
        d_test = reshape(next_d(i,:,:), n, 1);
        e_test = reshape(next_e(i,:,:), m, 1);
        next_L(i,:) = L(x, y, T_test, Hx, Hy, lam, mu, d_test, e_test);
    end
    [L_next, index] = min(next_L);
    Tx_next = reshape(next_T(index,:,:), n, d);
    d_next = reshape(next_d(index,:,:), n, 1);
    e_next = reshape(next_e(index,:,:), m, 1);

    while L_curr <= L_next && eta >= 1e-20
        eta = eta / 2;
        for i=1:num_combos
            if combos(i,1) == 1
                next_T(i,:,:) = Tx_curr - (eta * Tx_grad);
            else
                next_T(i,:,:) = Tx_curr;
            end
            if combos(i,2) == 1
                next_d(i,:,:) = d_curr - (eta * d_grad);
            else
                next_d(i,:,:) = d_curr;
            end
            if combos(i,3) == 1
                next_e(i,:,:) = e_curr - (eta * e_grad);
            else
                next_e(i,:,:) = e_curr;
            end
            T_test = reshape(next_T(i,:,:), n, d);
            d_test = reshape(next_d(i,:,:), n, 1);
            e_test = reshape(next_e(i,:,:), m, 1);
            next_L(i,:) = L(x, y, T_test, Hx, Hy, lam, mu, d_test, e_test);
        end
        [L_next, index] = min(next_L);
        Tx_next = reshape(next_T(index,:,:), n, d);
        d_next = reshape(next_d(index,:,:), n, 1);
        e_next = reshape(next_e(index,:,:), m, 1);
    end
end





% GRADIENT DESCENT
function [T_hist, L1_hist, L2_hist, L_hist, eta_hist, H_hist, iter, min_index, p_final, q_final] = grad_descent(x, y, eta, lam, mu, delta, epsilon)
    % INITIAL VALUES
    Tx = x;                                  % INITIAL MAP SHOULD BE THE ORIGINAL SET OF POINTS
    z = [Tx; y];                             % COMBINED SET OF POINTS - FOR BANDWIDTH
    Hy = bandwidth(y, length(x), true);      % BANDWIDTH FOR Y - USE ROBUST DEVIATION MEASURE
    Hz = bandwidth(z, length(x), true);      % BANDWIDTH FOR ALL POINTS - USE STD

    % INITIALIZING EMPTY HISTORY FOR ALL PLOTS OVER TIME
    T_hist = Tx; % FIRST ENTRY IN MAP HISTORY SHOULD BE THE SOURCE DISTRIBUTION
    H_hist = Hz;
    eta_hist = eta;
    L1_hist = C(x, Tx, delta);
    L2_hist = KL(Tx, y, Tx, Hz, Hy, delta, epsilon);
    L_hist = L1_hist + lam * L2_hist;

    criteria = 1e8; % ARBITRARY LARGE NUMBER TO START
    minimum = 1e8;
    iter = 1;
    min_index = 1;

    % CONTINUE UNTIL REACHING STOPPING CRITERIA
    while criteria > 0
    %while iter < 10
        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(iter, 1) == 0
            fprintf("Iteration: %d\n", iter)
        end

        if iter == 1000
            break
        end
        iter = iter+1;

        % GETTING NEW BANDWIDTH (CONVERGE TO HY)
        z = [Tx; y];
        Hz = bandwidth(z, length(x), true);
        Hz = (Hz + Hy) / 2;

        % GET NEW MAP TX AND LEARNING RATE ETA AT EACH STEP
        [eta, Tx, delta, epsilon] = adapt_learning(x, y, Tx, Hz, Hz, lam, mu, eta, delta, epsilon);

        % ADD CURRENT VALUES TO HISTORY DATA FOR PLOTTING
        T_hist(:,:,iter) = Tx;
        H_hist(:,:,iter) = Hz;
        eta_hist(iter,:) = eta;
        L1_hist(iter,:) = C(x, Tx, delta);
        L2_hist(iter,:) = KL(Tx, y, Tx, Hz, Hy, delta, epsilon);
        L_hist(iter,:) = L1_hist(iter,:) + lam*(L2_hist(iter,:));

        p_final = exp(delta)/sum(exp(delta));
        q_final = exp(epsilon)/sum(exp(epsilon));

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