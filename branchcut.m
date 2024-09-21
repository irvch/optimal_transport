% DISCRETE PARTIAL OPTIMAL TRANSPORT USING LINEAR PROGRAMMING

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")
mu_x = [0 1];
sigma_x = [0.25 0; 0 0.25];
X = mvnrnd(mu_x, sigma_x, 10);
%Y = X(3:end, :) + [0.5,-0.5];
Y = mvnrnd(mu_x, sigma_x, 8);

% LAMBDA CLOSER TO 0 --> EXTREME PARTIAL
lambda = 0.9;
%lambda = 0.7;
%lambda = 0.656; % STILL GETTING PARTIAL MASS IN P
%lambda = 0.6;
%lambda = 0.5;
%lambda = 0.36;
%lambda = 0.35914;
%lambda = 0.35;
%lambda = 0.2;
%lambda = 0.1;
%lambda = 0.01;
%lambda = 0.001;

% Define the distributions
p = ones(size(X, 1), 1);
p = p./sum(p); % NORMALIZED
q = ones(size(Y, 1), 1);
q = q./sum(q); % NORMALIZED

format long
[T, fval, p_new, q_new, alpha, beta] = branch(X, Y, p, q, lambda);
disp('Optimal transport plan')
%disp(T);
fprintf('Minimum Transportation Cost: %f\n', fval);
fprintf('Alpha: %f\n', alpha);
fprintf('Beta: %f\n', beta); 

disp('New mass at source points')
disp(p_new)
disp('New mass at target points')
disp(q_new)

% SET NULL VALUES TO NAN FOR PLOTTING
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
title("PARTIAL TRANSPORT MAP - 1ST RUN")
grid on;








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% BEGINNING CODE FOR BRANCH AND CUT METHOD FOR PARTIAL TRANSPORT

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


% INITIAL LINEAR PROGRAMMING CONSTRAINT MATRICES
function [f, Aeq, beq, A_ineq, b_ineq, lb, M, N] = constraints(X, Y, lambda, p, q)
    % COST MATRIX
    C = computeCostMatrix(X, Y);
    [N, M] = size(C);
    C_flat = reshape(C.',1,[]).';
    
    % FLATTEN COST INTO FORM FOR LINPROG, ADD ZEROS FOR A_NEW AND B_NEW,
    % ADD TWO LAMBDAS FOR PENALTY TERM
    f = [C_flat; zeros(N+M, 1); lambda; lambda];
    
    % EQUALITY CONSTRAINT Aeq
    Aeq_1 = [kron(eye(N), ones(1, M)); kron(ones(1, N), eye(M)); zeros(2, M*N)];
    Aeq_2 = [[diag(-1*ones(N+M, 1)); ones(1, N), zeros(1, M); zeros(1, N), ones(1, M)], zeros(M+N+2, 2)];
    Aeq = [Aeq_1, Aeq_2];

    % EQUALITY CONSTRAINT Beq
    beq = [zeros(N+M, 1); ones(2, 1)];

    % INEQUALITY CONSTRAINTS FOR P <= ALPHA*P_NEW, Q <= BETA*Q_NEW
    A_ineq = [zeros(N+M, M*N), diag(ones(N+M, 1)), [-1.*p(:); zeros(M, 1)], [zeros(N, 1); -1.*q(:)]];
    b_ineq = [zeros(N+M, 1)];
    
    % Lower bounds (non-negative constraints)
    lb = [zeros(N*M + N + M, 1); 1; 1];
end


% ADD NEW CONSTRAINT MATRICES FOR BRANCHES
function [Aeq, beq] = update(p, q, Aeq, beq, index, nudge, letter, M, N)
    % ADD NEW CONSTRAINT BASED ON BRANCH AT CERTAIN INDEX
    if ~isnan(index)
        new_constraint = zeros(1, M*N+M+N+2);
        if letter == "p" 
            % P_NEW = 0 CONSTRAINT
            new_constraint(index+M*N) = 1;
            if nudge == false 
                % P_NEW = ALPHA*P CONSTRAINT
                new_constraint(M*N+M+N+1) = -1*p(index);
            end
        elseif letter == "q"
            % Q_NEW = 0 CONSTRAINT
            new_constraint(index+M*N+N) = 1;
            if nudge == false
                % Q_NEW = BETA*Q CONSTRAINT
                new_constraint(M*N+M+N+2) = -1*q(index);
            end
        end
        % UPDATE CONSTRAINT MATRICES
        Aeq = [Aeq; new_constraint];
        beq = [beq; 0];
    end
end


% PARTIAL OPTIMAL TRANSPORT
function [T, fval, p_new, q_new, alpha, beta] = partialOT(f, A_ineq, b_ineq, Aeq, beq, lb, M, N)
    % SOLVE LINEAR PROGRAMMING
    options = optimoptions('linprog', 'Algorithm', 'dual-simplex');
    [solution, fval] = linprog(f, A_ineq, b_ineq, Aeq, beq, lb, [], options);

    % GET NEW T, P, Q, ALPHA, AND BETA FROM SOLN
    T = reshape(solution(1:N*M), [M, N]);
    p_new = solution(N*M+1:N*M+N);
    q_new = solution(N*M+N+1:N*M+N+M);
    alpha = solution(N*M+N+M+1);
    beta = solution(N*M+N+M+2);
end


% RETURN INDEX OF SMALLEST NON-ZERO VALUE
function [min_val, index] = find_branch(p_or_q)
    % BASE CASE IF NONE ARE PARTIAL
    index = nan;
    min_val = inf;
    
    % LOOP TO FIND WHERE VALUE IS NOT ALPHA*P_NEW OR BETA*Q_NEW
    max_val = max(p_or_q);
    for i=1:length(p_or_q)
        % IF MASS APPROX. ZERO OR DIFFERENCE IS NEGLIGIBLE, CONTINUE
        if p_or_q(i) < 1e-6 || abs(max_val - p_or_q(i)) < 1e-6
            continue;
        else
            if p_or_q(i) < min_val
                index = i;
                min_val = p_or_q(i);
            end
        end
    end
end


function [T_best, fval_best, p_best, q_best, alpha_best, beta_best] = branch(X, Y, p, q, lambda)
    % CREATE CONSTRAINT MATRICES
    [f, Aeq, beq, A_ineq, b_ineq, lb, M, N] = constraints(X, Y, lambda, p, q);

    % UPDATE CONSTRAINT MATRICES
    [Aeq, beq] = update(p, q, Aeq, beq, nan, nan, nan, M, N);

    % INITIAL RUN - AFTER INITIALIZING SO OUR BEST VALS NEVER HAVE PARTIAL MASSES
    [T, fval, p_new, q_new, alpha, beta] = partialOT(f, A_ineq, b_ineq, Aeq, beq, lb, M, N);
    disp("INITIAL PARTIAL TRANSPORT")
    disp([p, p_new])
    disp([q, q_new])

    % TRACK BEST VALUES
    T_best = T;
    fval_best = fval;
    p_best = p_new;
    q_best = q_new;
    alpha_best = alpha;
    beta_best = beta;

    % STACK TO STORE SUBPROBLEMS
    stack = {};

    % PICK WHETHER TO BRANCH P OR Q
    [p_val, p_index] = find_branch(p_new);
    [q_val, q_index] = find_branch(q_new);
    if ~isnan(p_index) || ~isnan(q_index)
        if p_val/max(p_new) >= q_val/max(q_new)
            letter = "q";
            index = q_index;
        else
            letter = "p";
            index = p_index;
        end
        
        % COMPUTE REGULAR OPTIMAL TRANSPORT COST SO NO PARTIAL RESULTS
        % BRANCH 1 - NUDGE TO ZERO
        [Aeq0, beq0] = update(p, q, Aeq, beq, index, true, letter, M, N);
        [T0, fval0, p0, q0, a0, b0] = partialOT(f, A_ineq, b_ineq, Aeq0, beq0, lb, M, N);
        disp([letter, "NUDGE TO ZERO", fval_best, fval0])
        disp([p_new, p0])
        disp([q_new, q0])
        newSubproblem1 = struct('T',T0, 'p',p0, 'q',q0, 'alpha',a0, 'beta',b0, 'fval',fval0, 'Aeq',Aeq0, 'beq',beq0);
        stack{end+1} = newSubproblem1;
        
        % BRANCH 2 - NUDGE TO ALPHA*P_NEW OR BETA*Q_NEW
        [Aeq1, beq1] = update(p, q, Aeq, beq, index, false, letter, M, N);
        [T1, fval1, p1, q1, a1, b1] = partialOT(f, A_ineq, b_ineq, Aeq1, beq1, lb, M, N);
        disp([letter, "NUDGE TO MAX", fval_best, fval1])
        disp([p_new, p1])
        disp([q_new, q1])
        newSubproblem2 = struct('T',T1, 'p',p1, 'q',q1, 'alpha',a1, 'beta',b1, 'fval',fval1, 'Aeq',Aeq1, 'beq',beq1);
        stack{end+1} = newSubproblem2;
    end

    % WHILE THERE ARE SUBPROBLEMS TO CHECK
    while ~isempty(stack)
        subproblem = stack{end};
        stack(end) = [];
        if subproblem.fval < fval_best
            T_best = subproblem.T;
            fval_best = subproblem.fval;
            p_best = subproblem.p;
            q_best = subproblem.q;
            alpha_best = subproblem.alpha;
            beta_best = subproblem.beta;
            Aeq_best = subproblem.Aeq;
            beq_best = subproblem.beq;

            %disp([fval_best, fval0, fval1])
            %disp([p_best, p0, p1])

            % CREATE NEW BRANCH OUT OF BEST MASSES
            [p_val, p_index] = find_branch(p_best);
            [q_val, q_index] = find_branch(q_best);

            if ~isnan(p_index) || ~isnan(q_index)
                if p_val/max(p_best) >= q_val/max(q_best)
                    letter = "q";
                    index = q_index;
                else
                    letter = "p";
                    index = p_index;
                end
                
                % BRANCH 1 - NUDGE TO ZERO
                [Aeq0, beq0] = update(p, q, Aeq_best, beq_best, index, true, letter, M, N);
                [T0, fval0, p0, q0, a0, b0] = partialOT(f, A_ineq, b_ineq, Aeq0, beq0, lb, M, N);
                disp([letter, "NUDGE TO ZERO", fval_best, fval0])
                disp([p_best, p0])
                disp([q_best, q0])
                newSubproblem1 = struct('T',T0, 'p_new',p0, 'q_new',q0, 'alpha',a0, 'beta',b0, 'fval',fval0, 'Aeq',Aeq0, 'beq',beq0);
                stack{end+1} = newSubproblem1;
                
                % BRANCH 2 - NUDGE TO ALPHA*P_NEW OR BETA*Q_NEW
                [Aeq1, beq1] = update(p, q, Aeq_best, beq_best, index, false, letter, M, N);
                [T1, fval1, p1, q1, a1, b1] = partialOT(f, A_ineq, b_ineq, Aeq1, beq1, lb, M, N);
                disp([letter, "NUDGE TO MAX", fval_best, fval1])
                disp([p_best, p1])
                disp([q_best, q1])
                newSubproblem2 = struct('T',T1, 'p_new',p1, 'q_new',q1, 'alpha',a1, 'beta',b1, 'fval',fval1, 'Aeq',Aeq1, 'beq',beq1);
                stack{end+1} = newSubproblem2;
            end
        end
    end
end