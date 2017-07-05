############################## HW1 Compuational Data Analysis ===================

############## Reading in dataset and installing some preliminary modules
#Set working directory
setwd("G:/Georgia Tech/Computational Data Analytics/Assignments")
#Loading packages
require(ggplot2)
require(data.table)
require(dplyr)
require(mvtnorm)
require(scales)


#Reading dataset
sem = fread("semeion.csv")
ncol(sem)
nrow(sem)
#1593 * 266 data table
#First 256 columns are the image data and the last 10 are the classification

images = subset(sem, select = c(1:256))

#Converting this to a matrix
#This will be 1593 x 256 matrix
X_data = data.matrix(data.frame(images))

#Clusters
clust_labels = subset(sem, select = c(257:266))

#Defining the dimensions and parameters in terms of variables
k = 10 #Number of clusters
n = 1593 #Number of data points
d = 256 #Number of Dimensions
q_list = c(0,2,4,6) #array of number of principal components
iterations = 40 #number of iterations


#Making the final lists to store the plotting variables
likelihood_list = vector(mode="list", length=length(q_list))
names(likelihood_list) = c("q0", "q2","q4","q6")

mu_list = vector(mode="list", length=length(q_list))
names(mu_list) = c("q0", "q2","q4","q6")

gamma_list = vector(mode="list", length=length(q_list))
names(gamma_list) = c("q0", "q2","q4","q6")

sigma_list = vector(mode="list", length=length(q_list))
names(sigma_list) = c("q0", "q2","q4","q6")

AIC_list = vector(length=length(q_list))


############################## Question 1 ================================
#Initializing the clusters
#K means clustering

#Will use 10 clusters as already known, 20 starting points for centers
# and 30 iterations for good convergence.
clust_model1 = kmeans(images, 10, iter.max = 30, nstart = 20)

#We have the initial cluster assignments
#We need to set Yik = 1 if it belongs to cluster k else 0
#Gamma matrix is going to hold the probabilities after soe iterations.
#It will have the probability that a data point (1593) will lie in a cluster.

#This is where we start the main loop of q's iterations

for (number in seq_along(q_list)){
q = q_list[number]  
print(" ")  
print(paste("Running loop wih q =", q))
#This is a n x k matrix - same as the labels matrix
gamma_matrix = matrix(0, nrow = nrow(clust_labels), ncol = ncol(clust_labels))

#Obtaining the cluster values
clusters = clust_model1$cluster

#Populating gamma matrix
for (i in 1:n){
  gamma_matrix[i, clusters[i]] = 1
}

#Next we need the matrix that will hold the centroid/mean of each cluster
#These means will also change as we iterate through the algorithm
#We will initialize it by the k-means centroids so that the estimate is good

#It is a k x d matrix (10 x 256)
mu_matrix = matrix(0, nrow = k, ncol = d)

#Mu matrix has to be calcuated by the matrix product of gamma with X_data and
#divided by Nk
#Initially it is equal to the centers given by the k-means
mu_matrix = clust_model1$centers

#We also need the pi vector which contains the proportion of points in a cluster
#We can get the proportion using the gamma matrix
pi = apply(gamma_matrix, 2, sum)/n

# Making the iterations list
likelihood <- rep(0, iterations)

# Initialization of covariance matrices
sigma <- array(dim=c(10,256,256))
px <- matrix(0, n, k)

################ Running the iteration loop with m and e step ==============

#M-step function
for (iter in 1:iterations){
  #We need a diagonal matrix with 1/Nk for each cluster.
  #This matrix gives us the inverse of the number of points in a cluster - 1/Nk
  point_composition_matrix = diag((pi*n)^-1)
  
  #Making a point composition vector
  point_composition_vector = (pi*n)^-1
  
  #Recalculating the mu_matrix
  mu_matrix = point_composition_matrix %*% t(gamma_matrix) %*% X_data
  
  #Recalculating pi
  pi = colSums(gamma_matrix)/n
  
  for(j in 1:k){
    #Initialize the covariance matrix
    cov_matrix = matrix(0,256,256)
    
    #Doing the spectral decomposition
    for(i in 1:n){
      cov_col = ((X_data[i,]-mu_matrix[j,]) %*% 
                          t((X_data[i,]-mu_matrix[j,]))) * gamma_matrix[i,j]
      cov_matrix = cov_matrix + cov_col
    }
    cov_matrix <- cov_matrix / sum(gamma_matrix[,j])
    #DOing the principal component analysis
    eigen_object = eigen(cov_matrix,symmetric=TRUE)
    
    #Computing the variance
    var_mat = sum(eigen_object$values[q+1:d], na.rm = T) / (d-q)
    
    #We need to put a condition on q = 0 as the sigma matrix fails to form
    if(q!=0) {
      prin_comp = eigen_object$vectors[,1:q]
      
      #Computing the diagonal eigen matrix 
      diag_eigen = diag(q)
      for(i in 1:q) {
        diag_eigen[i,i] = sqrt(eigen_object$values[i]-var_mat)
      }
      
      #Computing rank-q plus noise estimate 
      wq = prin_comp %*% diag_eigen
      #The sigma noise matrix is a 3 dimensional matrix
      sigma[j, , ] = wq %*% t(wq) + (var_mat * diag(d))
    
    }
    else {
      sigma[j, , ] = var_mat * diag(d)
    }
  }
  
  ################### Moving on to the E-step
  
  for(j in 1:k) {px[,j] = pi[j]*dmvnorm(X_data, mu_matrix[j,], sigma[j, , ], log = FALSE)}
  for(i in 1:n) {for(j in 1:k) { gamma_matrix[i,j] = px[i,j] / sum(px[i,]) }}
  
  ################### Calculating the likelihood
  
  likelihood[iter] <- sum(log(rowSums(px)))
  print(paste('The log-likelihood of loop no. ',iter,' is ',likelihood[iter]))
  
}

likelihood_list[[number]] = likelihood
gamma_list[[number]] = gamma_matrix
mu_list[[number]] = mu_matrix
sigma_list[[number]] = sigma

}


############### Plotting the various values of likelihood =================================

for (number in seq_along(q_list)) {
  
  final = as.data.frame(cbind(likelihood_list[[number]],c(1:iterations)))
  colnames(final) = c('ll','iteration')
  
  #Making the plots
  png(paste0('Log-likelihood for q = ',q_list[number],'.png'), width = 800, height = 500)
  
  #Using ggplot
  print(ggplot(final, aes(y=final$ll, x=final$iteration)) +
          geom_point() +
          labs(title = paste("Log-likelihood vs. Iteration for q =",q_list[number]),
               x = "Iteration", y = "Log-likelihood") +
          scale_x_discrete(limits=c(1:(iterations + 1))))
  dev.off()
}


########################### Calculating AIC to choose best q ==========================
for (number in seq_along(q_list)) {
  q = q_list[number]
  #calculate AIC
  AIC = -2*tail(likelihood_list[[number]],1) + 2*(d*q + 1 - (q*(q-1)/2))
  AIC_list[number] = AIC
  print(paste("The value of AIC for q = ", q, "is",round(AIC)))
}

print(paste('Best value of principal components should be chosen as',q_list[which.min(AIC_list)]))

########################## Visualizing the cluster ===================================
dev.new(width=7,height=3.5)
par(mai=c(0.05,0.05,0.05,0.05),mfrow=c(10,6))

for(i in 1:10){
  image(t(matrix(mu_list$q6[i,], byrow=TRUE,16,16)[16:1,]),col=gray(0:1),axes=FALSE)
  box()
  for(j in 1:5){
    tempX = rmvnorm(1, mean <- mu_list$q6[i,], sigma_list$q6[i,,])
    image(t(matrix(tempX, byrow=TRUE,16,16)[16:1,]),col=gray(0:1),axes=FALSE)
    box()
  }
}

########################## Accuracy assessment ========================================

# Alloting numbers to clusters
new_clusts = vector(length = n)

for(i in 1:n) {new_clusts[i] = which.max(gamma_list$q6[i,])}

#Getting the labels for the data points from the clust_labels
origLabel = apply(clust_labels,1,function(drow){return(which(drow=="1")-1)})

#Divinding the new labels according to the previous
clust_split = split(origLabel, new_clusts)
prop = lapply(clust_split, function(group){
  return(sort(table(group), decreasing=TRUE)[1])
})

# Initialize accuracy matrix
acc_matrix = matrix(0,4,10)

for(i in 1:10) {
  
  acc_matrix[1,i] = as.integer(names(prop[[i]]))
  acc_matrix[2,i] = as.integer(prop[[i]][[1]])
  acc_matrix[3,i] = as.integer(length(clust_split[[i]]))
  acc_matrix[4,i] = as.numeric(1 - (acc_matrix[2,i] / acc_matrix[3,i]))

}


acc_rate = 1 - sum(acc_matrix[2,]) / sum(acc_matrix[3,])

#Getting the mis classification rates
print("The mis-classification rates for each cluster are:")
print(percent(acc_matrix[4,]))
print(paste('The overall mis-classification rate is',percent(acc_rate)))

