############################## HW3 Compuational Data Analysis ===================

############## Reading in dataset and installing some preliminary modules
# Set working directory
setwd("G:/Georgia Tech/Computational Data Analytics/Assignments/HW3")
# Loading packages
require(ggplot2)
require(data.table)
require(dplyr)


# ----------------------- Gradient Descent for Multiple Linear Regression ---------------------

# Reading in the data
mlr = fread("MLR.csv")

# Dividing the data into X and Y
X = as.matrix(mlr[,1:30])
Y = as.matrix(mlr[,31])

########## The first step in our gradient descent algo is calculating the optimum step size

# Step size = Inverse of the largest eigen value of the Hessian of cost function

# calculating the Hessian = (Xt)X
hess = t(X) %*% X

# Getting the largest eigen value for this hessian matrix
max_eig = sort((eigen(hess))$values, decreasing = T)[1]

# Finally, learning rate = 1/largest eigen value
alpha = 1/max_eig


########## Writing the gradient descent algorithm
# Initializing parameters required for gradient descent

# Initializing the beta matrix from 0
beta = matrix(0, nrow = ncol(X), ncol = 1)
cost = c()
cost_old = 0 # Initializing cost function value from 0


# We also need a condition to stop the loop
# If the 2 norm of the newly calculated beta - old beta is smaller than a threshold
threshold = 0.00001
cost_differ = 1

while(cost_differ > threshold){
  cost_new = (norm((Y - X%*%beta), type = "2")**2)/(2*nrow(X))
  # Gradient descent iterations
  delta <- (t(X) %*% (X %*% beta - Y)) / length(Y)
  beta_new <- beta - alpha * delta
  
  cost_differ = abs(cost_new - cost_old)
  
  # Updating the old variables
  beta = beta_new
  cost = c(cost, cost_new)
  cost_old = cost_new
}

# Plotting the cost function at each iteration

plot(cost, type = "l",xlab = "Number of iterations", ylab = "Cost Function", 
     main = "Convergence of cost function", color = "blue")

# To calculate the difference between our calculated beta and true beta, we need to read in
# the true beta file

beta_true = fread("True_Beta.csv")
beta_true = as.matrix(beta_true)

# Calculating the mse

mse = (norm((beta_true - beta), type = "2")**2)/30
cat("The mse between the calculated beta and the true beta is ", mse)


# ------------- Stochastic Gradient Descent for Multiple Linear Regression -------------------

# In stochastic, we update the gradient based on one statistical unit.

# Updating the step size - we need algo to go slower as we are approximating gradient
alpha = 1/max_eig
alpha = alpha/nrow(X)

# Initializing the variables again
beta = matrix(0, nrow = ncol(X), ncol = 1)
cost = c()
cost_old = 0

cost_differ = 1

while(cost_differ > threshold){
  stoch_cost = c()
  # We need to calculate the gradient and update beta at each observation instead of the
  # updating it after calculating the gradient on the whole dataset
  for (i in seq(nrow(X))){
    cost_new <- sum((X[i,]%*%beta- Y[i])^2)/2
    
    delta <- (X[i,] %*% (X[i,]%*%beta - Y[i]))
    beta_new <- beta - (alpha * delta)
    
    
    beta <- beta_new
    # Storing each cost calculated
    stoch_cost[i] <- cost_new
  }
  
  # Loop stop condition
  cost_differ = abs(mean(stoch_cost) - cost_old)
  
  cost <- c(cost, mean(stoch_cost))
  
  cost_old <- mean(stoch_cost)
}

# Plotting out the results after each run
plot(cost, type = "l",xlab = "Number of iterations", ylab = "Cost Function", 
     main = "Convergence of cost function - Stochastic GD")

# Calculating the mse again

mse = (norm((beta_true - beta), type = "2")**2)/30
cat("The mse between the calculated beta and the true beta is ", mse)


# ------------------------- Mini-batch Gradient Descent -------------------------------------

# In Batch GD, we approximate the gradient not at a single observation but at a group of observations.

# Updating the step size according to the value of the batch size
b = 100

alpha = b/(nrow(X) * max_eig)

# Initializing the variables again
beta = matrix(0, nrow = ncol(X), ncol = 1)
cost = c()
cost_old = 0

cost_differ = 1

while(cost_differ > threshold){
  stoch_cost = c()
  # We need to calculate the gradient and update beta at each batch instead of the
  # updating it after calculating the gradient on the whole dataset
  for (i in seq(nrow(X)/b)){
    batch = sample(nrow(X), size = b, replace=FALSE)
    
    cost_new <- sum((X[batch, ]%*%beta- Y[batch])^2)/(2*length(Y[batch]))
    
    delta <- t(X[batch, ]) %*% (X[batch, ]%*%beta - Y[batch])
    beta_new <- beta - (alpha * delta * 1/b)
    
    
    beta <- beta_new
    # Storing each cost calculated
    stoch_cost[i] <- cost_new
  }
  
  # Loop stop condition
  cost_differ = abs(mean(stoch_cost) - cost_old)
  
  cost <- c(cost, mean(stoch_cost))
  
  cost_old <- mean(stoch_cost)
}

# Plotting out the results after each run
plot(cost, type = "l",xlab = "Number of iterations", ylab = "Cost Function", 
     main = "Convergence of cost function - Batch = 100 Stochastic GD")

# Calculating the mse again

mse = (norm((beta_true - beta), type = "2")**2)/30
cat("The mse between the calculated beta and the true beta is ", mse)

# I have used the above code for b = 10, 25 and 100.


# ------------- Online Principal Components Analysis -------------

# Reading in the data
opca <- fread("OPCA.csv")

true_eig = fread("True_eigvector.csv")
true_eig = as.matrix(true_eig)

# First, we will try Oja's algrithm with fixed step size

# Initializing the parameters
step = 0.01
d = 20

Wi = as.matrix(rep(1/sqrt(d),d))

# Oja's algorithm
dist = c()

for (i in seq(nrow(opca)/d)) { 
  
  Ai = as.matrix(opca[(d*(i-1)+1):(d*(i)), ])
  
  Wi = Wi + step*(Ai %*% Wi) 
  Wi = Wi/norm(Wi, "2")
  
  dist[i] = 1- (t(Wi)%*%(true_eig))**2
  
}

# plotting the distances

plot(dist, type = "l", main = "Dist(wi,v) against number of iterations", xlab = "Number of iterations", ylab = "Distance between W and V")


# NOw we will use varying step size

# Initializing the parameters

d = 20
Wi = as.matrix(rep(1/sqrt(d),d))
dist = c()

# Oja's algorithm

for (i in seq(nrow(opca)/d)) { 
  
  step = 1/(100+i)
  
  Ai = as.matrix(opca[(d*(i-1)+1):(d*(i)),]) 
  
  Wi = Wi + step*(Ai %*% Wi) 
  Wi = Wi/norm(Wi,"2") 
  
  dist[i] = 1- (t(Wi)%*%(true_eig))**2
  
}

# plotting the distances

plot(dist, type = "l", main = "Dist(wi,v) against number of iterations", xlab = "Number of iterations", ylab = "Distance between W and V")

# --------------------------------- End --------------------------------

