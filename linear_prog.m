% Define the supply, demand, and cost matrix
supply = [15; 20; 15];
x = supply./sum(supply);
demand = [10; 20; 20; 5];
y = demand./sum(demand);

% Define penalty parameters
k = 0; % Penalty coefficient
a = 8; % Value of a
b = 5; % Value of b


% Define the cost matrix to use in linprog
[c, penalty_value] = compute_cost_matrix(x, y, k, a, b); 
disp('Cost matrix with penalty:');
disp(c);
disp(['Penalty value: ', num2str(penalty_value)]);

% Define the constraint matrices
A_eq = kron(eye(numel(x)), ones(1, numel(y))); % Equality constraint matrix
b_eq = x; % Equality constraint vector
A_ineq = kron(ones(1, numel(x)), eye(numel(y))); % Inequality constraint matrix
b_ineq = y; % Inequality constraint vector

% Solve the linear programming problem
[x, fval, exitflag, output] = linprog(c, A_ineq, b_ineq, A_eq, b_eq, zeros(size(c)));

% Reshape the solution vector x to obtain the transportation matrix
transport_matrix = reshape(x, [3, 4]);

% Display the results
fprintf('Optimal transportation cost: %.2f\n', fval);
disp('Optimal transportation matrix:');
disp(transport_matrix);

function [cost_matrix_with_penalty, penalty_value] = compute_cost_matrix(suppliers, demanders, k, a, b)
    num_suppliers = size(suppliers, 1);
    num_demanders = size(demanders, 1);
    
    % Initialize the cost matrix
    cost_matrix = zeros(num_suppliers, num_demanders);
    
    % Compute the L2 norm between each supplier and demander
    for i = 1:num_suppliers
        for j = 1:num_demanders
            cost_matrix(i, j) = norm(suppliers(i, :) - demanders(j, :));
        end
    end
    
    % Apply penalty term k(a + b)
    penalty_value = k * (a + b);
    cost_matrix_with_penalty = cost_matrix + penalty_value;
end