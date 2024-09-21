% DISCRETE PARTIAL OPTIMAL TRANSPORT USING LINEAR PROGRAMMING

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")
mu_x = [0 1];
sigma_x = [0.25 0; 0 0.25];
X = mvnrnd(mu_x, sigma_x, 10);
%Y = X(3:end, :) + [0.5,-0.5];
Y = mvnrnd(mu_x, sigma_x, 8);

% LAMBDA CLOSER TO 0 --> EXTREME PARTIAL
%lambda = 0.7;
lambda = 0.656;
lambda = 0.6;
%lambda = 0.5;
%lambda = 0.36;
%lambda = 0.35914;
%lambda = 0.35;
%lambda = 0.2;
%lambda = 0.1;
%lambda = 0.01;

% Define the distributions
p = ones(size(X, 1), 1);
p = p./sum(p); % NORMALIZED
q = ones(size(Y, 1), 1);
q = q./sum(q); % NORMALIZED

format long
[T, fval, p_new, q_new, alpha, beta] = partialOT(X, Y, p, q, lambda);
disp('Optimal transport plan')
disp(T);
fprintf('Minimum Transportation Cost: %f\n', fval);
fprintf('Alpha: %f\n', alpha);
fprintf('Beta: %f\n', beta); 

%[T, fval] = solveOptimalTransport(X, Y, p, q);
%disp('Optimal transport plan')
%disp(T);
%fprintf('Minimum Transportation Cost: %f\n', fval);

disp('New mass at source points')
disp(p_new)
disp('New mass at target points')
disp(q_new)

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
axis([-1.5 3 -0.5 3])
scatter(X(:,1), X(:,2), p*500, 'filled', 'blue');
scatter(Y(:,1), Y(:,2), q*500, 'filled', 'red');
legend('Source Pts', 'Target Pts');
title("INITIAL DATA POINTS")
grid on;
hold off;

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

% PLOTTING NEW DATA
figure();
hold on;
axis([-1.5 3 -0.5 3])
scatter(X(:,1), X(:,2), p*500, 'filled', 'blue');
scatter(X(:,1), X(:,2), p_new*500, 'filled', 'green');
scatter(Y(:,1), Y(:,2), q*500, 'filled', 'red');
scatter(Y(:,1), Y(:,2), q_new*500, 'filled', 'magenta');
% Draw arrows from X to Y based on the transport matrix
for i = 1:size(X,1)
    for j = 1:size(Y,1)
        if T(j,i) >= 1e-6  % Draw an arrow if the transport amount is greater than zero
            quiver(X(i,1), X(i,2), Y(j,1) - X(i,1), Y(j,2) - X(i,2), 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
        end
    end
end
legend('Leftover Source Pts', 'Subsampled Source Pts', 'Leftover Target Pts', 'Subsampled Target Pts');
title("PARTIAL TRANSPORT MAP")
grid on;

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

% PARTIAL OPTIMAL TRANSPORT
function [T, fval, p_new, q_new, alpha, beta] = partialOT(X, Y, p, q, lambda)

    % COST MATRIX
    C = computeCostMatrix(X, Y);
    [N, M] = size(C);
    %C_flat = C(:);
    C_flat = reshape(C.',1,[]).';
    
    % FLATTEN COST INTO FORM FOR LINPROG, ADD ZEROS FOR A_NEW AND B_NEW,
    % ADD TWO LAMBDAS FOR PENALTY TERM
    f = [C_flat; zeros(N+M, 1); lambda; lambda];
    
    % EQUALITY CONSTRAINT Aeq
    Aeq_1 = [kron(eye(N), ones(1, M)); kron(ones(1, N), eye(M)); zeros(2, M*N)];
    Aeq_2 = [[diag(-1*ones(N+M, 1)); ones(1, N), zeros(1, M); zeros(1, N), ones(1, M)], zeros(M+N+2, 2)];
    Aeq = [Aeq_1, Aeq_2; zeros(2, M*N+M+N+2)];

    % EQUALITY CONSTRAINT Beq
    beq = [zeros(N+M, 1); ones(2, 1); 0; 0];
    
    % INEQUALITY CONSTRAINTS FOR P <= ALPHA*P_NEW, Q <= BETA*Q_NEW
    A_ineq = [zeros(N+M, M*N), diag(ones(N+M, 1)), [-1.*p(:); zeros(M, 1)], [zeros(N, 1); -1.*q(:)]];
    b_ineq = [zeros(N+M, 1)];
    
    % Lower bounds (non-negative constraints)
    lb = [zeros(N*M + N + M, 1); 1; 1];
    
    % Solve the linear programming problem
    options = optimoptions('linprog', 'Algorithm', 'dual-simplex');
    [solution, fval] = linprog(f, A_ineq, b_ineq, Aeq, beq, lb, [], options);

    % Extract T, a_new, and b_new from solution
    T = reshape(solution(1:N*M), [M, N]);
    p_new = solution(N*M+1:N*M+N);
    q_new = solution(N*M+N+1:N*M+N+M);
    alpha = solution(N*M+N+M+1);
    beta = solution(N*M+N+M+2);
end

%{
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
%}