% DISCRETE PARTIAL OPTIMAL TRANSPORT USING IMPLICIT MINIMAX

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")
mu_x = [0 1];
sigma_x = [0.25 0; 0 0.25];

X = mvnrnd(mu_x, sigma_x, 30);
Y = X + [0.05, -0.05];
addition = mvnrnd(mu_x, sigma_x*0.25, 5) + [2.5, -2];
Y = [Y; addition];

% LAMBDA CLOSER TO 0 --> EXTREME PARTIAL
% IF ITERATION IS HIGH ENOUGH, LAMBDA IS ESSENTIALLY IGNORED (HMM)
lambda = 0.001;
eta = 5;
maxIter = 25;

% Define the distributions
p = ones(size(X, 1), 1);
p = p./sum(p); % NORMALIZED
q = ones(size(Y, 1), 1);
q = q./sum(q); % NORMALIZED

format long
[T, p_new, q_new, alpha, beta] = partialOT(X, Y, eta, lambda, maxIter);
disp('Optimal transport plan')
disp(T);
fprintf('Alpha: %f\n', alpha);
fprintf('Beta: %f\n', beta);

disp('New mass at source points')
disp(p_new)
disp('New mass at target points')
disp(q_new)

% PLOTTING ORIGINAL DATA
figure();
hold on;
%axis([-1.5 3 -0.5 3])
scatter(X(:,1), X(:,2), p*500, 'filled', 'blue');
scatter(Y(:,1), Y(:,2), q*500, 'filled', 'red');
legend('Source Pts', 'Target Pts');
title("INITIAL DATA POINTS")
grid on;
hold off;

% PLOTTING NEW DATA
figure();
hold on;
%axis([-1.5 3 -0.5 3])
scatter(X(:,1), X(:,2), p*500, 'filled', 'blue');
scatter(X(:,1), X(:,2), p_new*500, 'filled', 'green');
scatter(Y(:,1), Y(:,2), q*500, 'filled', 'red');
scatter(Y(:,1), Y(:,2), q_new*500, 'filled', 'magenta');
% Draw arrows from X to Y based on the transport matrix
for i = 1:size(X,1)
    for j = 1:size(Y,1)
        if T(j,i) >= 1e-8  % Draw an arrow if the transport amount is greater than zero
            quiver(X(i,1), X(i,2), Y(j,1) - X(i,1), Y(j,2) - X(i,2), 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
        end
    end
end
legend('Leftover Source Pts', 'Subsampled Source Pts', 'Leftover Target Pts', 'Subsampled Target Pts');
title("PARTIAL TRANSPORT MAP")
grid on;

%% TRANSPORT HELPER FUNCTIONS
function [A_eq, A_ineq, b_eq, b_ineq] = constraints(N, M, p, q)
    % NORMALIZE P AND Q
    p = p./sum(p);
    q = q./sum(q);

    % EQUALITY CONSTRAINT Aeq
    Aeq_1 = [kron(eye(N), ones(1, M)); kron(ones(1, N), eye(M)); zeros(2, M*N)];
    Aeq_2 = [[diag(-1*ones(N+M, 1)); ones(1, N), zeros(1, M); zeros(1, N), ones(1, M)], zeros(M+N+2, 2)];
    A_eq = [Aeq_1, Aeq_2];

    % EQUALITY CONSTRAINT Beq
    b_eq = [zeros(N+M, 1); ones(2, 1)];


    % INEQUALITY CONSTRAINTS FOR P <= ALPHA*P_NEW, Q <= BETA*Q_NEW
    A_ineq = [zeros(N+M, M*N), diag(ones(N+M, 1)), [-1.*p(:); zeros(M, 1)], [zeros(N, 1); -1.*q(:)]];
    b_ineq = [zeros(N+M, 1)];

    % TO ENFORCE ALPHA AND BETA >= 1
    lb_alpha = [zeros(1, N*M + N + M), -1, 0];
    lb_beta = [zeros(1, N*M + N + M), 0, -1];
    
    % WANT AX <= b: MAKE NEGATIVE
    A_ineq = [A_ineq; lb_alpha; lb_beta];
    b_ineq = [b_ineq; -1; -1];
end

function L = lagrangian(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq)
    L = c'*x.^2 + y_eq'*(A_eq*x.^2 - b_eq) + y_ineq.^2'*(A_ineq*x.^2 - b_ineq);
end

function G = gradient(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq)
    % FIRST ORDER PARTIAL DERIVATIVES OF LAGRANGIAN

    Lx = 2 * (c + A_eq'*y_eq + A_ineq'*(y_ineq.^2)) .* x;
    Ly_eq = (A_eq*x.^2 - b_eq);
    Ly_ineq = 2 * (A_ineq*x.^2 - b_ineq) .* y_ineq;

    % GRADIENT MATRIX
    G = [Lx; Ly_eq; Ly_ineq];
end

function H = hessian(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq)
    y = y_eq;
    w = y_ineq;

    % SECOND ORDER PARTIAL DERIVATIVE MATRICES
    Lxx = 2 * diag(c + A_eq'*y + A_ineq'*(w.^2));
    Lxy = 2 * diag(x) * A_eq';
    Lxw = 4 * diag(x) * A_ineq' * diag(w);

    Lyx = 2 * A_eq * diag(x);
    Lyy = zeros(length(y));
    Lyw = zeros(length(y));

    Lwx = 4 * diag(w) * A_ineq * diag(x);
    Lwy = zeros(length(w));
    Lww = 2 * diag(A_ineq*x.^2 - b_ineq);

    % HESSIAN
    H = [Lxx, Lxy, Lxw; 
        Lyx, Lyy, Lyw; 
        Lwx, Lwy, Lww];
end

%{
function Z_next = grad_descent(c, Z_curr, A_eq, A_ineq, b_eq, b_ineq, eta, N, M)
    % COMPUTE NEW X AND Y VIA IMPLICIT TWISTED GRAD DESCENT
    x_curr = Z_curr(1:M*N + N + M + 2);
    y_curr = Z_curr(M*N + N + M + 3:end);
    y_eq = y_curr(1:N+M+2);
    y_ineq = y_curr(N+M+3:end);

    J = [eye(length(x_curr)), zeros(length(x_curr), length(y_curr));
        zeros(length(y_curr), length(x_curr)), -1*eye(length(y_curr))];

    G = gradient(c, x_curr, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq);
    H = hessian(c, x_curr, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq);
    Z_next = Z_curr - eta*(J + eta * H) \ G;
    %Z_next = Z_curr - eta * J * G;
end
%}

function Z_next = grad_descent(c, Z_curr, A_eq, A_ineq, b_eq, b_ineq, eta, N, M)
    % COMPUTE NEW X AND Y VIA IMPLICIT TWISTED GRAD DESCENT
    x_curr = Z_curr(1:M*N + N + M + 2);
    y_curr = Z_curr(M*N + N + M + 3:end);
    y_ineq_curr = y_curr(1:N+M+2);
    y_eq_curr = y_curr(N+M+3:end);

    J = blkdiag(eye(length(x_curr)), -eye(length(y_curr)));
    G = gradient(c, x_curr, y_eq_curr, y_ineq_curr, A_eq, A_ineq, b_eq, b_ineq);
    H = hessian(c, x_curr, y_eq_curr, y_ineq_curr, A_eq, A_ineq, b_eq, b_ineq);
    L_curr = lagrangian(c, x_curr, y_eq_curr, y_ineq_curr, A_eq, A_ineq, b_eq, b_ineq);

    % BACKTRACKING LINE SEARCH FOR LEARNING RATE TUNING
    % SET INITIAL LEARNING RATE TO OLD ETA * CONSTANT FOR GROWTH
    Z_next = Z_curr - eta * J * G;
    x_next = Z_next(1:M*N + N + M + 2);

    y_next = Z_next(M*N + N + M + 3:end);
    y_ineq_next = y_next(1:N+M+2);
    y_eq_next = y_next(N+M+3:end);

    Lx_next = lagrangian(c, x_next, y_eq_curr, y_ineq_curr, A_eq, A_ineq, b_eq, b_ineq);
    Ly_next = lagrangian(c, x_curr, y_eq_next, y_ineq_next, A_eq, A_ineq, b_eq, b_ineq);

    % ITERATIVELY SHRINK LEARNING RATE BY HALF
    const = 1e-3;
    while Lx_next < L_curr + const*eta*norm(G, 2)^2 && Ly_next > L_curr + const*eta*norm(G, 2)^2
        eta = eta / 2;
        Z_next = Z_curr - eta * J * G;
        x_next = Z_next(1:M*N + N + M + 2);

        y_next = Z_next(M*N + N + M + 3:end);
        y_ineq_next = y_next(1:N+M+2);
        y_eq_next = y_next(N+M+3:end);
    
        Lx_next = lagrangian(c, x_next, y_eq_curr, y_ineq_curr, A_eq, A_ineq, b_eq, b_ineq);
        Ly_next = lagrangian(c, x_curr, y_eq_next, y_ineq_next, A_eq, A_ineq, b_eq, b_ineq);
    end

    % UPDATE
    Z_next = Z_curr - eta*(J + eta * H) \ G;
end


%% PARTIAL OPTIMAL TRANSPORT
function [T, p_new, q_new, alpha, beta] = partialOT(x, y, eta, lambda, maxIter)
    % COST MATRIX
    C = pdist2(x, y);

    % LENGTH OF SOURCE AND TARGET RESPECTIVELY
    [N, M] = size(C);
    
    % FLATTEN COST INTO FORM FOR LINPROG, ADD ZEROS FOR A_NEW AND B_NEW,
    % ADD TWO LAMBDAS FOR PENALTY TERM
    C_flat = reshape(C.',1,[]).';
    c = [C_flat; zeros(N+M, 1); lambda; lambda];

    % MINIMAX OVER THESE
    T_init = (ones(M*N, 1)/(M*N)).^1/2;
    p_init = (ones(N, 1)/N).^2;
    q_init = (ones(M, 1)/M).^2;
    alpha = 1;
    beta = 1;

    x = [T_init; p_init; q_init; alpha; beta];
    y = [ones(2*N + 2*M + 4, 1)]/(2*M + 2*M + 4);
    %y = rand(2*N + 2*M + 4, 1);
    %y = (0.4*y).^1/2;
    y_eq = y(1:N+M+2);
    y_ineq = y(N+M+3:end);
    
    Z = [x; y];

    % CONSTRAINT MATRIXES A^t*X >= B
    [A_eq, A_ineq, b_eq, b_ineq] = constraints(N, M, p_init, q_init);

    % FOR TRACKING HISTORY OVER ITERATIONS
    %T_hist = reshape(x(1:N*M), [M, N]);
    %eta_hist = eta;
    L_hist = lagrangian(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq);
    
    criteria = 1e10; % ARBITRARY LARGE NUMBER TO START
    iter = 1;
    %min_index = 1;

    while criteria > 0

        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(iter, 100) == 0
            fprintf("Iteration: %d\n", iter)
        end

        if iter == maxIter
            break
        end
        iter = iter+1;
    
        Z = grad_descent(c, Z, A_eq, A_ineq, b_eq, b_ineq, eta, N, M);
    
        x = Z(1:M*N + N + M + 2);
        y = Z(M*N + N + M + 3:end);
        y_eq = y(1:N+M+2);
        y_ineq = y(N+M+3:end);

        % TRACKING HISTORY
        %T_hist(:,:,iter) = T;
        %eta_hist(iter,:) = eta;
        L_hist(iter,:) = lagrangian(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq);

    end
    % GETTING RELEVANT VALUES FROM X
    X = x.^2;
    T = reshape(X(1:N*M), [M, N]);
    p_new = X(N*M+1:N*M+N);
    p_new = p_new./sum(p_new);
    q_new = X(N*M+N+1:N*M+N+M);
    q_new = q_new./sum(q_new);
    alpha = X(N*M+N+M+1);
    beta = X(N*M+N+M+2);

    T(T <= 1e-6) = 0;
    p_new = p_new(p_new > 1e-6);
    q_new = q_new(q_new > 1e-6);
end



