% DISCRETE PARTIAL OPTIMAL TRANSPORT USING BRANCH & CUT ON IMPLICIT MINIMAX
% ALONG WITH THRESHOLDING

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")
mu_x = [0 1];
sigma_x = [0.25 0; 0 0.25];
source = mvnrnd(mu_x, sigma_x, 10);
target = mvnrnd(mu_x, sigma_x, 8);

%{
theta = 90; % to rotate counterclockwise
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
target = target*R;

% SOURCE DATA - TWO MOONS
N_half = 20; % Number of points per semi-circle
theta = linspace(0, pi, N_half)'; % Angles for semicircle

source_upper = [cos(theta), sin(theta)] * 0.5;
source_lower = [cos(theta) + 1, sin(theta) - 0.5]* 0.5 * [cosd(180) -sind(180); sind(180) cosd(180)];
source = [source_upper; source_lower] + [0.25, 0];
source = source + 0.02 * randn(size(source));

% TARGET DATA
target = source * [cosd(45) -sind(45); sind(45) cosd(45)];
addition = mvnrnd([1, -1], [0.01, 0; 0, 0.01], 5);
target = [target; addition];
%}


% LAMBDA CLOSER TO 0 --> EXTREME PARTIAL
% IF ITERATION IS HIGH ENOUGH, LAMBDA IS ESSENTIALLY IGNORED (HMM)
lambda = 0.01;
epsilon = 1/(max(length(source), length(target))) * 1/4;
%epsilon = 0.01;
eta = 6;
iters = 50;


% Define the distributions
p = ones(size(source, 1), 1);
p = p./sum(p); % NORMALIZED
q = ones(size(target, 1), 1);
q = q./sum(q); % NORMALIZED

% PLOT ORIGINAL DATA
figure();
hold on;
scatter(source(:,1), source(:,2), p*1000, 'filled', 'blue');
scatter(target(:,1), target(:,2), q*1000, 'filled', 'red');
legend('Source Pts', 'Target Pts');
title("INITIAL DATA POINTS")
grid on;
hold off;

% START TIMER FOR ALGORITHM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic

format long
[T, c, p_new, q_new, alpha, beta] = branch(source, target, p, q, eta, lambda, epsilon, iters);

T(T < 0.005) = 0;
T = T./sum(T, "all");

p_new(p_new <= epsilon) = 0;
q_new(q_new <= epsilon) = 0;


% END TIMER
runtime = toc;

fprintf('Optimal transport plan: total cost = %f\n', c)
%disp(T);
fprintf('Alpha: %f\n', alpha);
fprintf('Beta: %f\n', beta);

disp('New mass at source points')
disp(p_new)
disp('New mass at target points')
disp(q_new)

% DISPLAY RUNTIME
disp(['ALGO RUNTIME: ' num2str(runtime) ' sec'])


% PLOT ORIGINAL DATA
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

%{
figure();
hold on;
scatter(source(:,1), source(:,2), p*1000, 'filled', 'blue');
scatter(target(:,1), target(:,2), q*1000, 'filled', 'red');
legend('Source Pts', 'Target Pts');
title("INITIAL DATA POINTS")
grid on;
hold off;
%}

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
scatter(source(:,1), source(:,2), p*1000, 'filled', 'blue');
scatter(source(:,1), source(:,2), p_new*1000, 'filled', 'green');
scatter(target(:,1), target(:,2), q*1000, 'filled', 'red');
scatter(target(:,1), target(:,2), q_new*1000, 'filled', 'magenta');
% Draw arrows from X to Y based on the transport matrix
for i = 1:size(source,1)
    for j = 1:size(target,1)
        if T(j,i) >= 1e-6  % Draw an arrow if the transport amount is greater than zero
            quiver(source(i,1), source(i,2), ...
                target(j,1) - source(i,1), target(j,2) - source(i,2), ...
                0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
        end
    end
end
legend('Leftover Source Pts', 'Subsampled Source Pts', 'Leftover Target Pts', 'Subsampled Target Pts');
title("PARTIAL TRANSPORT MAP - THRESHOLDING")
grid on;


%% TRANSPORT HELPER FUNCTIONS
function mass = normalize(mass)
    % Normalize probability mass distribution
    mass = mass / sum(mass);
end

function [c, y, A_eq, A_ineq, b_eq, b_ineq, M, N] = constraints(source, target, lambda, p, q)
    % COST MATRIX
    C = pdist2(source, target);
    [N, M] = size(C);
    
    % FLATTEN COST INTO FORM FOR LINPROG, ADD ZEROS FOR A_NEW AND B_NEW,
    % ADD TWO LAMBDAS FOR PENALTY TERM
    C_flat = reshape(C.',1,[]).';
    c = [C_flat; zeros(N+M, 1); lambda; lambda];

    % NORMALIZE P AND Q
    p = normalize(p);
    q = normalize(q);

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

    y = [ones(2*N + 2*M + 4, 1)]/(2*M + 2*M + 4);
end

function L = lagrangian(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq)
    L = c'*x.^2 + y_eq'*(A_eq*x.^2 - b_eq) + y_ineq.^2'*(A_ineq*x.^2 - b_ineq);
end

%% FOR GRADIENT DESCENT
function G = gradient(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq)
    % FIRST ORDER PARTIAL DERIVATIVES OF LAGRANGIAN
    Lx = 2 * (c + A_eq'*y_eq + A_ineq'*(y_ineq.^2)) .* x;
    Ly_eq = (A_eq*x.^2 - b_eq);
    Ly_ineq = 2 * (A_ineq*x.^2 - b_ineq) .* y_ineq;

    % GRADIENT MATRIX
    G = [Lx; Ly_ineq; Ly_eq];
end

function H = hessian(c, x, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq)
    w = y_ineq;
    y = y_eq;

    % SECOND ORDER PARTIAL DERIVATIVE MATRICES
    Lxx = 2 * diag(c + A_eq'*y + A_ineq'*(w.^2));
    Lxy = 2 * diag(x) * A_eq';
    Lxw = 4 * diag(x) * A_ineq' * diag(w);

    Lyx = 2 * A_eq * diag(x);
    Lyy = zeros(length(y));
    Lyw = zeros(length(y), length(w));

    Lwx = 4 * diag(w) * A_ineq * diag(x);
    Lwy = zeros(length(w), length(y));
    Lww = 2 * diag(A_ineq*x.^2 - b_ineq);

    % HESSIAN
    H = [Lxx, Lxw, Lxy; 
        Lwx, Lww, Lwy;
        Lyx, Lyw, Lyy];
end

function Z_next = grad_descent(c, Z_curr, A_eq, A_ineq, b_eq, b_ineq, eta, N, M)
    % COMPUTE NEW X AND Y VIA IMPLICIT TWISTED GRAD DESCENT
    x_curr = Z_curr(1:M*N + N + M + 2);
    y_curr = Z_curr(M*N + N + M + 3:end);
    y_ineq = y_curr(1:N+M+2);
    y_eq = y_curr(N+M+3:end);

    %J = blkdiag(eye(length(x_curr)), -eye(length(y_curr)));

    J = [eye(length(x_curr)), zeros(length(x_curr), length(y_curr));
        zeros(length(y_curr), length(x_curr)), -1*eye(length(y_curr))];

    G = gradient(c, x_curr, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq);
    H = hessian(c, x_curr, y_eq, y_ineq, A_eq, A_ineq, b_eq, b_ineq);
    Z_next = Z_curr - eta*(J + eta * H) \ G;
end


%% REGULAR OPTIMAL TRANSPORT
function [T, fval] = OT(X, Y, p, q)
    % Compute the cost matrix using the Frobenius norm
    C = pdist2(X, Y);
    
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


%% PARTIAL OPTIMAL TRANSPORT
function [T, cval, p_new, q_new, alpha, beta] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters)  

    % MINIMAX OVER THESE
    T_init = (ones(M*N, 1)/(M*N)).^1/2;
    p_init = p.^2;
    q_init = q.^2;
    alpha = 1;
    beta = 1;

    x = [T_init; p_init; q_init; alpha; beta];
    Z = [x; y];

    iter = 1;
    while iter < iters
        Z = grad_descent(c, Z, Aeq, A_ineq, beq, b_ineq, eta, N, M);
        x = Z(1:M*N + N + M + 2);
        iter = iter+1;
    end

    % GETTING RELEVANT VALUES FROM X
    X = x.^2;
    cval = c'*X;
    T = reshape(X(1:N*M), [M, N]);
    
    p_new = normalize(X(N*M+1:N*M+N));
    q_new = normalize(X(N*M+N+1:N*M+N+M));

    alpha = X(N*M+N+M+1);
    beta = X(N*M+N+M+2);
end

%% HELPER FUNCTIONS FOR BRANCH AND CUT
% ADD NEW CONSTRAINT MATRICES FOR BRANCHES
function [p, q] = update(p, q, index, nudge, letter)
    % COMPLETELY REMOVE CONSTRAINT BASED ON BRANCH AT CERTAIN INDEX
    if letter == "p" 
        % P_NEW = 0 CONSTRAINT
        p(index) = 0;
        if nudge == false
            % P_NEW = ALPHA*P CONSTRAINT
            p(index) = max(p);
        end
    elseif letter == "q"
        % Q_NEW = 0 CONSTRAINT
        q(index) = 0;
        if nudge == false
            % Q_NEW = BETA*Q CONSTRAINT
            q(index) = max(q);
        end
    end
    p = p./sum(p);
    q = q./sum(q);
end

% CHANGE MASS TO ZERO AT CERTAIN INDEX
function [p_or_q_original, seen] = threshold(p_or_q, p_or_q_next, epsilon, letter, seen)
    % UNIFORM MASS
    nonzero = 0;
    for i = 1:length(p_or_q)
        if p_or_q(i) ~= 0
            nonzero = nonzero + 1;
        end
    end
    unif = 1/nonzero;

    % LOOP AND UPDATE SEEN VALUES
    for i = 1:length(p_or_q)
        if any(cellfun(@(x) isequal(x, [letter, i]), seen))
            continue;
        else
            if p_or_q_next(i) <= epsilon
                p_or_q(i) = 0;
                seen{end+1} = [letter, i];
            elseif p_or_q_next(i) >= unif - epsilon
                p_or_q(i) = unif;
                seen{end+1} = [letter, i];
            end
        end
    end
    p_or_q_original = p_or_q./sum(p_or_q);
end

% FIX VALUE TO ZERO OR MAX IF NOT ALREADY DONE - TO HANDLE EDGE CASE
function p_or_q_update = fix(p_or_q, p_or_q_next, epsilon, index)
    nonzero = 0;
    for i = 1:length(p_or_q)
        if p_or_q(i) ~= 0
            nonzero = nonzero + 1;
        end
    end
    unif = 1/nonzero;

    value = p_or_q_next(index);
    if abs(value - epsilon) < abs(unif - epsilon - value)
        p_or_q(index) = 0;
    else
        p_or_q(index) = unif;
    end
    p_or_q_update = p_or_q./sum(p_or_q);
end


% RETURN INDEX OF SMALLEST NON-ZERO VALUE
function [min_val, index] = find_branch(p_or_q, letter, seen, epsilon)
    % BASE CASE IF NONE ARE PARTIAL
    index = nan;
    min_val = inf;
    
    % LOOP TO FIND WHERE VALUE IS NOT WITHIN THRESHOLD VALUES
    for i=1:length(p_or_q)
        % IF MASS APPROX. ZERO OR PAIR HAS ALREADY BEEN CONSTRAINED, CONTINUE
        if p_or_q(i) == 0
            continue;
        % IF INDEX HAS BEEN SEEN ALREADY
        elseif any(cellfun(@(x) isequal(x, [letter, i]), seen))
            fprintf("IGNORING %s, INDEX %d\n", letter, i)
            continue;
        elseif p_or_q(i) < min_val && p_or_q(i) > epsilon 
            index = i;
            min_val = p_or_q(i);
        end
    end
end

%% FINAL BRANCH AND CUT PROCEDURE
function [T_best, cval_best, p_best, q_best, alpha_best, beta_best] = branch(source, target, p, q, eta, lambda, epsilon, iters)
    % CREATE CONSTRAINT MATRICES
    [c, y, Aeq, A_ineq, beq, b_ineq, M, N] = constraints(source, target, lambda, p, q);

    % INITIAL RUN
    [~, ~, p_new, q_new, ~, ~] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters);
    
    disp("INITIAL PARTIAL TRANSPORT")
    disp([p, p_new])
    disp([q, q_new])

    % TRACK COMBOS OF INDEX AND LETTER THAT HAVE BEEN SET OR NUDGED TO MAX OR ZERO
    seen = {};

    % APPLY INITIAL THRESHOLD
    [p, seen] = threshold(p, p_new, epsilon, "p", seen);
    [q, seen] = threshold(q, q_new, epsilon, "q", seen);
    [T, cval, p_new, q_new, alpha, beta] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters);
    
    disp("AFTER INITIAL THRESHOLDING")
    disp([p, p_new])
    disp([q, q_new])

    % TRACK BEST VALUES
    valid_solns = {};

    % STACK TO STORE SUBPROBLEMS
    stack = {};

    % PICK WHETHER TO BRANCH P OR Q
    [p_val, p_index] = find_branch(p_new, "p", seen, epsilon);
    [q_val, q_index] = find_branch(q_new, "q", seen, epsilon);

    if ~isnan(p_index) || ~isnan(q_index)

        if p_val/max(p_new) >= q_val/max(q_new)
            letter = "q";
            index = q_index;
        else
            letter = "p";
            index = p_index;
        end
        seen{end+1} = [letter, index];

        % COMPUTE REGULAR OPTIMAL TRANSPORT COST SO NO PARTIAL RESULTS
        % BRANCH 1 - NUDGE TO ZERO
        [p, q] = update(p, q, index, true, letter);
        [T0, cval0, p0, q0, a0, b0] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters);
        fprintf("%s NUDGE INDEX %d TO ZERO: COST = %f\n", letter, index, cval0)

        [p_update0, seen] = threshold(p, p0, epsilon, "p", seen);
        [q_update0, seen] = threshold(q, q0, epsilon, "q", seen);
        %disp([p_new, p, p0])
        %disp([q_new, q, q0])
        
        % BRANCH 2 - NUDGE TO ALPHA*P_NEW OR BETA*Q_NEW
        [p, q] = update(p, q, index, false, letter);
        [T1, cval1, p1, q1, a1, b1] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters);
        fprintf("%s NUDGE INDEX %d TO MAX: COST = %f\n", letter, index, cval1)

        [p_update1, seen] = threshold(p, p1, epsilon, "p", seen);
        [q_update1, seen] = threshold(q, q1, epsilon, "q", seen);
        %disp([p_new, p, p1])
        %disp([q_new, q, q1])

        if cval0 <= cval1
            % FIX TO ZERO - IF NOT ALREADY DONE
            if letter == "p"
                p_update0 = fix(p_update0, p0, epsilon, index);
            elseif letter == "q"
                q_update0 = fix(q_update0, q0, epsilon, index);
            end
            newSubproblem0 = struct('T',T0, 'y', y, 'p_new',p0, 'q_new',q0, ...
                                    'alpha',a0, 'beta',b0, 'cval',cval0, ...
                                    'Aeq',Aeq, 'beq',beq, ...
                                    'p',p_update0, 'q',q_update0);
            stack{end+1} = newSubproblem0;
        else
            % FIX TO MAX - IF NOT ALREADY DONE
            if letter == "p"
                p_update1 = fix(p_update1, p1, epsilon, index);
            elseif letter == "q"
                q_update1 = fix(q_update1, q1, epsilon, index);
            end
            newSubproblem1 = struct('T',T1, 'y', y, 'p_new',p1, 'q_new',q1, ...
                                    'alpha',a1, 'beta',b1, 'cval',cval1, ...
                                    'Aeq',Aeq, 'beq',beq, ...
                                    'p',p_update1, 'q',q_update1);
            stack{end+1} = newSubproblem1;
        end

    else
        valid_solns{end+1} = struct('T',T, 'y', y, 'p_new',p_new, 'q_new',q_new, ...
                                    'alpha',alpha, 'beta',beta, 'cval',cval, ...
                                    'Aeq',Aeq, 'beq',beq, 'p',p, 'q',q);
    end

    % WHILE THERE ARE SUBPROBLEMS TO CHECK
    while ~isempty(stack)
        subproblem = stack{end};
        stack(end) = [];
        p_new = subproblem.p_new;
        q_new = subproblem.q_new;
        Aeq_next = subproblem.Aeq;
        beq_next = subproblem.beq;
        y_next = subproblem.y;
        p = subproblem.p;
        q = subproblem.q;

        % CREATE NEW BRANCH OUT OF BEST MASSES
        [p_val, p_index] = find_branch(p_new, "p", seen, epsilon);
        [q_val, q_index] = find_branch(q_new, "q", seen, epsilon);
    
        if ~isnan(p_index) || ~isnan(q_index)
            for i=1:length(seen)
                pair = seen{i};
                if isequal(["p", p_index], pair) || isequal(["q", q_index], pair)
                    continue
                elseif isequal(["p", p_index], pair) && ~isequal(["q", q_index], pair)
                    letter = "q";
                    index = q_index;
                elseif ~isequal(["p", p_index], pair) && isequal(["q", q_index], pair)
                    letter = "p";
                    index = p_index;
                else
                    if p_val/max(p_new) >= q_val/max(q_new)
                        letter = "q";
                        index = q_index;
                    else
                        letter = "p";
                        index = p_index;
                    end
                end
            end
            seen{end+1} = [letter, index];
            fprintf("ADDING %s, INDEX %d\n", letter, index)

            % BRANCH 1 - NUDGE TO ZERO
            [p, q] = update(p, q, index, true, letter);
            [T0, cval0, p0, q0, a0, b0] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters);
            fprintf("%s NUDGE INDEX %d TO ZERO: COST = %f\n", letter, index, cval0)

            [p_update0, seen] = threshold(p, p0, epsilon, "p", seen);
            [q_update0, seen] = threshold(q, q0, epsilon, "q", seen);
            %disp([p_new, p, p0])
            %disp([q_new, q, q0])
            
            % BRANCH 2 - NUDGE TO ALPHA*P_NEW OR BETA*Q_NEW
            [p, q] = update(p, q, index, false, letter);
            [T1, cval1, p1, q1, a1, b1] = partialOT(c, y, A_ineq, b_ineq, Aeq, beq, p, q, eta, M, N, iters);
            fprintf("%s NUDGE INDEX %d TO MAX: COST = %f\n", letter, index, cval1)

            [p_update1, seen] = threshold(p, p1, epsilon, "p", seen);
            [q_update1, seen] = threshold(q, q1, epsilon, "q", seen);
            %disp([p_new, p, p1])
            %disp([q_new, q, q1])
            
            % AFTER THRESHOLDING, THEN WHICHEVER INDEX IS NUDGED, MUST FIX
            % IT TO DIRECTION OF LEAST COST
            if cval0 <= cval1
                % FIX TO ZERO - IF NOT ALREADY DONE
                if letter == "p"
                    p_update0 = fix(p_update0, p0, epsilon, index);
                elseif letter == "q"
                    q_update0 = fix(q_update0, q0, epsilon, index);
                end
                newSubproblem0 = struct('T',T0, 'y', y, 'p_new',p0, 'q_new',q0, ...
                                        'alpha',a0, 'beta',b0, 'cval',cval0, ...
                                        'Aeq',Aeq, 'beq',beq, ...
                                        'p',p_update0, 'q',q_update0);
                stack{end+1} = newSubproblem0;
            else
                % FIX TO MAX - IF NOT ALREADY DONE
                if letter == "p"
                    p_update1 = fix(p_update1, p1, epsilon, index);
                elseif letter == "q"
                    q_update1 = fix(q_update1, q1, epsilon, index);
                end
                newSubproblem1 = struct('T',T1, 'y', y, 'p_new',p1, 'q_new',q1, ...
                                        'alpha',a1, 'beta',b1, 'cval',cval1, ...
                                        'Aeq',Aeq, 'beq',beq, ...
                                        'p',p_update1, 'q',q_update1);
                stack{end+1} = newSubproblem1;
            end

        else
            % FIX ONE LAST EDGE CASE THAT FIXES BRANCH CUT POINTS TO A SIDE
            % PREVENTS UNENDING LOOP
            valid_solns{end+1} = subproblem;
        end
    end
    
    c_min = inf;
    min_index = inf;
    % FIND MINIMUM COST OF ALL VALID SOLNS - CORRESPONDS TO BEST SOLN
    disp("VALID SOLUTIONS")
    disp(valid_solns)
    for i = 1:length(valid_solns)
        if valid_solns{i}.cval < c_min
            c_min = valid_solns{i}.cval;
            min_index = i;
        end
    end
    % RETURN VALUES OF THE OPTIMAL SOLUTION
    optimal_soln = valid_solns{min_index};
    T_best = optimal_soln.T;
    cval_best = optimal_soln.cval;
    p_best = optimal_soln.p;
    q_best = optimal_soln.q;
    alpha_best = optimal_soln.alpha;
    beta_best = optimal_soln.beta;

    [T_best, cval_best] = OT(source, target, p_best, q_best);

end

