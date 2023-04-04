# Optimal Transport

Program for multidimensional optimal transport between data sets.  
Uses kernel density estimation to approximate discrete systems of points into probability density functions.  
Global cost function takes the form of the penalty method - using a normalized Frobenius norm as the distance metric and a constraint function measuring the difference between pdfs based around map T(x) and y.  
Linearly increasing lambda in global cost serves a function similar to Lagrange multipliers.  
Applying gradient descent with an adaptive learning rate to guarantee convergence to global minimum.  