# Optimal Transport

Uses kernel density estimation to approximate discrete systems of points into probability density functions.  
Global cost function takes the form of the penalty method.  
Uses a normalized Frobenius norm as the distance metric.  
The constraint function measures the difference between the pdfs based around the map T(x) and y.  
Linearly increasing lambda in the global cost function serves a role similar to Lagrange multipliers.  
Applying gradient descent with an adaptive learning rate to guarantee convergence to global minimum.  
