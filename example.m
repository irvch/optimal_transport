% DISCRETE PARTIAL OPTIMAL TRANSPORT USING LINEAR PROGRAMMING

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")

eta = 0.0005;
iters = 100;
x = [1; 1];
y = [1; 1];

format long
[x1_hist, x2_hist, y1_hist, y2_hist, L_hist] = partialOT(x, y, eta, iters);
disp("Printing results")
fprintf("X1: %4.2f --- Actual: 0.4 \n", x1_hist(end))
fprintf("X2: %4.2f --- Actual: 1.8 \n", x2_hist(end))
fprintf("Y1: %4.2f --- Actual: 0.6 \n", y1_hist(end))
fprintf("Y1: %4.2f --- Actual: 0.8 \n", y2_hist(end))
fprintf("Lagrangian: %4.2 --- Actual: 4.8 f", L_hist(end))

figure;
hold on;
plot(1:iters, x1_hist, "-")
plot(1:iters, y2_hist, "-")
plot(1:iters, y1_hist, "-")
plot(1:iters, y2_hist, "-")
plot(1:iters, L_hist, "-")
hold off;
legend("x1", "x2", "y1", "y2", "L");

%[solution, fval] = linprog(x, A, b, lb, [], options);

% COST MATRIX USING FROBENIUS NORM

function L = lagrangian(c, x, y, A, b)
    L = c'*x.^2 + y.^2'*(A*x.^2 - b);
    %L = (x - 0.5)*(y - 0.5) + 1/3*exp(-(x-0.5)^2-(y-0.75)^2);
end

function G = gradient(c, x, y, A, b)
    % FIRST ORDER PARTIAL DERIVATIVES OF LAGRANGIAN
    Lx = 2 * (c + A'*(y.^2)) .* x;
    Ly = 2 * (A*x.^2 - b) .* y;

    % GRADIENT MATRIX
    G = [Lx; Ly];
end

function H = hessian(c, x, y, A, b)
    % SECOND ORDER PARTIAL DERIVATIVE MATRICES
    Lxx = 2 * diag(c + A'*(y.^2));
    Lxy = 4 * diag(x) * A' * diag(y);

    Lyx = 4 * diag(y) * A * diag(x);
    Lyy = 2 * diag(A*x.^2 - b);

    % HESSIAN
    H = [Lxx, Lxy;
        Lyx, Lyy];
end

function Z_next = grad_descent(c, Z_curr, A, b, eta)
    % COMPUTE NEW X AND Y VIA IMPLICIT TWISTED GRAD DESCENT
    x_curr = Z_curr(1:2);
    y_curr = Z_curr(3:end);

    J = [eye(length(x_curr)), zeros(length(x_curr), length(y_curr));
        zeros(length(y_curr), length(x_curr)), -1*eye(length(y_curr))];

    G = gradient(c, x_curr, y_curr, A, b);
    H = hessian(c, x_curr, y_curr, A, b);
    
    %alpha = 10;
    %mu_max = 100;
    %mu_next = min(alpha*mu_curr, mu_max);
    
    %eta = eta / norm(G, 'fro');
    %disp(G)
    disp(norm(G, 'fro'))

    %Z_next = Z_curr - eta*(J + eta * H) \ G;

    %x_next = Z_next(1:2);
    %y_next = Z_next(3:end);

    %L_left = lagrangian(c, x_next, y_curr, A, b)
    %L_next = lagrangian(c, x_next, y_next, A, b)
    %L_right = lagrangian(c, x_curr, y_next, A, b)
    
    Z_next = Z_curr - eta * J * G;
end

% PARTIAL OPTIMAL TRANSPORT
function [x1_hist, x2_hist, y1_hist, y2_hist, L_hist] = partialOT(x, y, eta, iters)
    Z = [x; y];
    c = [3; 2];
    A = [1, 2; 3, 1];
    b = [4; 3];
    x1_hist = x(1)^2;
    x2_hist = x(2)^2;
    y1_hist = y(1)^2;
    y2_hist = y(2)^2;
    L_hist = lagrangian(c, x, y, A, b);
    
    criteria = 1e10; % ARBITRARY LARGE NUMBER TO START
    iter = 1;

    while criteria > 0

        % FOR KEEPING TRACK OF ITERATION PROGRESS
        if mod(iter, 100) == 0
            fprintf("Iteration: %d\n", iter)
        end
        if iter == iters
            break
        end
        iter = iter+1;
    
        Z = grad_descent(c, Z, A, b, eta);    
        x = Z(1:2);
        y = Z(3:end);
        x1_hist(iter,:) = x(1)^2;
        x2_hist(iter,:) = x(2)^2;
        y1_hist(iter,:) = y(1)^2;
        y2_hist(iter,:) = y(2)^2;
        L_hist(iter,:) = lagrangian(c, x, y, A, b);
    end
end

