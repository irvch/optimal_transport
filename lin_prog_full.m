% DISCRETE OPTIMAL TRANSPORT USING LINEAR PROGRAMMING

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")
mu_x = [0 1];
sigma_x = [0.25 0; 0 0.25];
X = mvnrnd(mu_x, sigma_x, 20);
%Y = X(3:end, :) + [0.5,-0.5];
Y = mvnrnd(mu_x, sigma_x, 20);

X = mvnrnd(mu_x, sigma_x, 50);
Y = X + [0.05, -0.05];
addition = mvnrnd(mu_x, sigma_x*0.25, 5) + [3, -4];
Y = [Y; addition];

%{
% PYRAMID DATA SETS
Y = table2array(readtable('revised data 3.xlsx', Sheet='every'));
X = table2array(readtable('revised data set 1.xlsx', Sheet='every'));
%X = table2array(readtable('revised data set 1.xlsx', Sheet='every (3)'));

% SOURCE DATA - TWO MOONS
N_half = 20; % Number of points per semi-circle
theta = linspace(0, pi, N_half)'; % Angles for semicircle

source_upper = [cos(theta), sin(theta)] * 0.5;
source_lower = [cos(theta) + 1, sin(theta) - 0.5]* 0.5 * [cosd(180) -sind(180); sind(180) cosd(180)];
source = [source_upper; source_lower] + [0.25, 0];
X = source + 0.02 * randn(size(source));

% TARGET DATA
target = X * [cosd(45) -sind(45); sind(45) cosd(45)];
addition = mvnrnd([1, -1], [0.01, 0; 0, 0.01], 5);
Y = [target; addition];
%}

% Define the distributions
p = ones(size(X, 1), 1);
p = p./sum(p); % NORMALIZED
q = ones(size(Y, 1), 1);
q = q./sum(q); % NORMALIZED

maxIter = 100;
%tol = 1e-5;
%epsilon = 0.01;

format long
%[T, fval] = OT_Sinkhorn(X, Y, p, q, epsilon, maxIter, tol);
[T, fval] = OT(X, Y, p, q);
disp('Optimal transport plan')
disp(size(T));
fprintf('Minimum Transportation Cost: %f\n', fval);

disp('New mass at source points')
disp(size(p))
disp('New mass at target points')
disp(size(q))

for i=1:length(p)
    if p(i,:) <= 1e-6
        p(i,:) = nan;
    end
end

for i=1:length(q)
    if q(i,:) <= 1e-6
        q(i,:) = nan;
    end
end

% PLOTTING ORIGINAL DATA
figure();
hold on;
%axis([-1.5 3 -0.5 3])
scatter(X(:,1), X(:,2), p*500, 'filled', 'blue');
scatter(Y(:,1), Y(:,2), q*500, 'filled', 'red');
%scatter3(X(:,1), X(:,2), X(:,3), 'filled', 'blue')
%scatter3(Y(:,1), Y(:,2), Y(:,3), 'filled', 'red')
legend('Source Pts', 'Target Pts');
title("INITIAL DATA POINTS")
grid on;
hold off;

%{
for i=1:length(p_new)
    if p_new(i,:) <= 1e-6
        p_new(i,:) = nan;
    end
end

for i=1:length(q_new)
    if q_new(i,:) <= 1e-6
        q_new(i,:) = nan;
    end
end
%}

% PLOTTING NEW DATA
figure();
hold on;
%axis([-1.5 3 -0.5 3])
%scatter(X(:,1), X(:,2), p*500, 'filled', 'blue');
%scatter(X(:,1), X(:,2), p_new*500, 'filled', 'green');
%scatter(Y(:,1), Y(:,2), q*500, 'filled', 'red');
%scatter(Y(:,1), Y(:,2), q_new*500, 'filled', 'magenta');

scatter(X(:,1), X(:,2), 'filled', 'blue');
scatter(X(:,1), X(:,2), 'filled', 'green');
scatter(Y(:,1), Y(:,2), 'filled', 'red');
scatter(Y(:,1), Y(:,2), 'filled', 'magenta');

%scatter3(X(:,1), X(:,2), X(:,3), 'filled', 'blue')
%scatter3(Y(:,1), Y(:,2), Y(:,3), 'filled', 'red')

% Draw arrows from X to Y based on the transport matrix
[M, N] = size(T);  % M = number of targets (rows in Y), N = number of sources (rows in X)
for i = 1:min(N, size(X, 1))  % Make sure we do not exceed the number of source points
    for j = 1:min(M, size(Y, 1))  % Make sure we do not exceed the number of target points
        if T(j,i) >= 1e-6  % Draw an arrow if the transport amount is greater than zero
            quiver(X(i,1), X(i,2), Y(j,1) - X(i,1), Y(j,2) - X(i,2), 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
            %quiver3(X(i,1), X(i,2), X(i,3), Y(j,1) - X(i,1), Y(j,2) - X(i,2), Y(j,3) - X(i,3),  0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
        end
    end
end

legend('Leftover Source Pts', 'Subsampled Source Pts', 'Leftover Target Pts', 'Subsampled Target Pts');
title("PARTIAL TRANSPORT MAP")
grid on;
hold off;

% COST MATRIX USING FROBENIUS NORM
function C = computeCostMatrix(X, Y)
    N = size(X, 1);
    M = size(Y, 1);
    C = zeros(N, M);
    for i = 1:N
        for j = 1:M
            C(i, j) = norm(X(i,:) - Y(j,:));
        end
    end
end

function [T, fval] = OT_Sinkhorn(X, Y, p, q, epsilon, maxIter, tol)
    % Compute the cost matrix using the Frobenius norm
    C = computeCostMatrix(X, Y);
    
    % Exponentiate the negative cost matrix scaled by epsilon (regularization)
    K = exp(-C / epsilon);
    
    % Initialize scaling vectors (u and v for Sinkhorn iterations)
    u = ones(size(p));
    v = ones(size(q));
    
    % Main loop: Iterate until convergence or max number of iterations is reached
    for iter = 1:maxIter
        u_prev = u;
        
        % Update u and v iteratively
        u = p ./ (K * v);
        v = q ./ (K' * u);
        
        % Check convergence (relative change in scaling vectors)
        if max(abs(u - u_prev)) < tol
            break;
        end
    end
    
    % Compute the optimal transport matrix
    T = diag(u) * K * diag(v);
    
    % Compute the value of the regularized transport problem (Sinkhorn distance)
    fval = sum(sum(T .* C));
end

%  Solve the optimal transport problem
function [T, fval] = OT(X, Y, p, q)
    % Compute the cost matrix using the Frobenius norm
    C = computeCostMatrix(X, Y);

    % Number of sources and targets
    [N, M] = size(C);

    % Objective function: Flatten the cost matrix C to a vector
    C_flat = reshape(C.',1,[]).';
    f = C_flat;

    % Equality constraints: Ensure the mass balance
    Aeq = [kron(eye(N), ones(1, M)); kron(ones(1, N), eye(M))];
    beq = [p; q];

    % Lower bounds for the decision variables (non-negative constraints)
    lb = zeros(N * M, 1);

    % Solve the linear programming problem
    options = optimoptions('linprog', 'Algorithm', 'dual-simplex');
    [Tvec, fval] = linprog(f, [], [], Aeq, beq, lb, [], options);

    % Reshape the solution vector back to a matrix to match the problem structure
    T = reshape(Tvec, [M, N]);
end