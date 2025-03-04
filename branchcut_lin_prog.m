% DISCRETE PARTIAL OPTIMAL TRANSPORT USING BRANCH & CUT ON LINEAR PROGRAMMING

% DISCRETE SOURCE AND TARGET COORDINATES
rng("default")
mu_x = [0 1];
sigma_x = [0.25 0; 0 0.25];
%X = mvnrnd(mu_x, sigma_x, 10);
%Y = X(3:end, :) + [0.5,-0.5];
%Y = mvnrnd(mu_x, sigma_x, 8);

%X = mvnrnd(mu_x, sigma_x, 250);
%Y = mvnrnd(mu_x, sigma_x, 250);
%Y = X + [0.05, -0.05];
%addition = mvnrnd(mu_x, sigma_x, 50) + [2, -2];
%Y = [Y; addition];

X = table2array(readtable('revised data set 1.xlsx', Sheet='every'));
Y = table2array(readtable('revised data 3.xlsx', Sheet='every'));


N_half = 40; % Number of points per semi-circle
theta = linspace(0, pi, N_half)'; % Angles for semicircle

% SOURCE DATA - TWO MOONS
source_upper = [cos(theta), sin(theta)] * 0.5;
source_lower = [cos(theta) + 1, sin(theta) - 0.5]* 0.5 * [cosd(180) -sind(180); sind(180) cosd(180)] - 0.25;
source = [source_upper; source_lower] + [0.375, 0];
source = source + 0.02 * randn(size(source));

% TARGET DATA
target = source * [cosd(45) -sind(45); sind(45) cosd(45)];
addition = mvnrnd([0, -1], [0.005, 0; 0, 0.005], 20);
target = [target; addition];


% LAMBDA CLOSER TO 0 --> EXTREME PARTIAL
% IF ITERATION IS HIGH ENOUGH, LAMBDA IS ESSENTIALLY IGNORED (HMM)
lambda = 0.2;
%epsilon = 1/(2*min(length(source), length(target)));
epsilon = 0.01;
eta = 2.5;
iters = 200;

% Define the distributions
p = ones(size(source, 1), 1);
p = p./sum(p); % NORMALIZED
q = ones(size(target, 1), 1);
q = q./sum(q); % NORMALIZED

format long
[T, fval, p_new, q_new, alpha, beta] = branch(source, target, p, q, lambda);
disp('Optimal transport plan')
%disp(T);
fprintf('Minimum Transportation Cost: %f\n', fval);
fprintf('Alpha: %f\n', alpha);
fprintf('Beta: %f\n', beta); 

disp('New mass at source points')
disp(size(p_new))
disp('New mass at target points')
disp(size(q_new))

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
%axis([-1.5 3 -0.5 3])
scatter(source(:,1), source(:,2), p*500, 'filled', 'blue');
scatter(target(:,1), target(:,2), q*500, 'filled', 'red');

%scatter3(X(:,1), X(:,2), X(:,3), 'filled', 'blue')
%scatter3(Y(:,1), Y(:,2), Y(:,3), 'filled', 'red')
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
%axis([-1.5 3 -0.5 3])
scatter(source(:,1), source(:,2), p*500, 'filled', 'blue');
scatter(source(:,1), source(:,2), p_new*500, 'filled', 'green');
scatter(target(:,1), target(:,2), q*500, 'filled', 'red');
scatter(target(:,1), target(:,2), q_new*500, 'filled', 'magenta');

%scatter3(X(:,1), X(:,2), X(:,3), p*10000, 'filled', 'blue')
%scatter3(X(:,1)+0.01, X(:,2), X(:,3), p_new*10000, 'filled', 'green')
%scatter3(Y(:,1), Y(:,2), Y(:,3), q*10000, 'filled', 'red')
%scatter3(Y(:,1), Y(:,2), Y(:,3), q_new*10000, 'filled', 'magenta')


% Draw arrows from X to Y based on the transport matrix
for i = 1:size(source,1)
    for j = 1:size(target,1)
        if T(j,i) >= 1e-6  % Draw an arrow if the transport amount is greater than zero
            quiver(source(i,1), source(i,2), target(j,1) - source(i,1), target(j,2) - source(i,2), 0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
            %quiver3(X(i,1), X(i,2), X(i,3), Y(j,1) - X(i,1), Y(j,2) - X(i,2), Y(j,3) - X(i,3),  0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
        end
    end
end
%legend('Leftover Source Pts', 'Subsampled Source Pts', 'Leftover Target Pts', 'Subsampled Target Pts');
title("PARTIAL TRANSPORT MAP - BRANCH & CUT")
grid on;
hold off;







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
    %disp([p, p_new])
    %disp([q, q_new])

    % TRACK BEST VALUES
    valid_solns = {};

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
        disp([letter, index, "NUDGE TO ZERO", fval0])
        disp([p_new, p0])
        disp([q_new, q0])
        newSubproblem1 = struct('T',T0, 'p',p0, 'q',q0, 'alpha',a0, 'beta',b0, 'fval',fval0, 'Aeq',Aeq0, 'beq',beq0);
        stack{end+1} = newSubproblem1;
        
        % BRANCH 2 - NUDGE TO ALPHA*P_NEW OR BETA*Q_NEW
        [Aeq1, beq1] = update(p, q, Aeq, beq, index, false, letter, M, N);
        [T1, fval1, p1, q1, a1, b1] = partialOT(f, A_ineq, b_ineq, Aeq1, beq1, lb, M, N);
        disp([letter, index, "NUDGE TO MAX", fval1])
        disp([p_new, p1])
        disp([q_new, q1])
        newSubproblem2 = struct('T',T1, 'p',p1, 'q',q1, 'alpha',a1, 'beta',b1, 'fval',fval1, 'Aeq',Aeq1, 'beq',beq1);
        stack{end+1} = newSubproblem2;
    else
        valid_solns{end+1} = struct('T',T, 'p',p_new, 'q',q_new, 'alpha',alpha, 'beta',beta, 'fval',fval, 'Aeq',Aeq, 'beq',beq);
    end

    % WHILE THERE ARE SUBPROBLEMS TO CHECK
    while ~isempty(stack)
        subproblem = stack{end};
        stack(end) = [];
        p_next = subproblem.p;
        q_next = subproblem.q;
        Aeq_next = subproblem.Aeq;
        beq_next = subproblem.beq;

        %disp(size(Aeq))
        %disp(size(Aeq_next))

        % CREATE NEW BRANCH OUT OF BEST MASSES
        [p_val, p_index] = find_branch(p_next);
        [q_val, q_index] = find_branch(q_next);

        if ~isnan(p_index) || ~isnan(q_index)
            if p_val/max(p_next) >= q_val/max(q_next)
                letter = "q";
                index = q_index;
            else
                letter = "p";
                index = p_index;
            end
            
            % BRANCH 1 - NUDGE TO ZERO
            [Aeq0, beq0] = update(p, q, Aeq_next, beq_next, index, true, letter, M, N);
            [T0, fval0, p0, q0, a0, b0] = partialOT(f, A_ineq, b_ineq, Aeq0, beq0, lb, M, N);
            disp([letter, index, "NUDGE TO ZERO", fval0])
            disp([p_next, p0])
            disp([q_next, q0])
            newSubproblem1 = struct('T',T0, 'p',p0, 'q',q0, 'alpha',a0, 'beta',b0, 'fval',fval0, 'Aeq',Aeq0, 'beq',beq0);
            stack{end+1} = newSubproblem1;
                
            % BRANCH 2 - NUDGE TO ALPHA*P_NEW OR BETA*Q_NEW
            [Aeq1, beq1] = update(p, q, Aeq_next, beq_next, index, false, letter, M, N);
            [T1, fval1, p1, q1, a1, b1] = partialOT(f, A_ineq, b_ineq, Aeq1, beq1, lb, M, N);
            disp([letter, index, "NUDGE TO MAX", fval1])
            disp([p_next, p1])
            disp([q_next, q1])
            newSubproblem2 = struct('T',T1, 'p',p1, 'q',q1, 'alpha',a1, 'beta',b1, 'fval',fval1, 'Aeq',Aeq1, 'beq',beq1);
            stack{end+1} = newSubproblem2;
        else
            valid_solns{end+1} = subproblem;
        end
    end
    
    f_min = inf;
    min_index = inf;
    % FIND MINIMUM COST OF ALL VALID SOLNS - CORRESPONDS TO BEST SOLN
    for i = 1:length(valid_solns)
        if valid_solns{i}.fval < f_min
            f_min = valid_solns{i}.fval;
            min_index = i;
        end
    end
    % RETURN VALUES OF THE OPTIMAL SOLUTION
    optimal_soln = valid_solns{min_index};
    T_best = optimal_soln.T;
    fval_best = optimal_soln.fval;
    p_best = optimal_soln.p;
    q_best = optimal_soln.q;
    alpha_best = optimal_soln.alpha;
    beta_best = optimal_soln.beta;
end


