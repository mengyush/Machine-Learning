Machine Learning: Programming Exercise 1
=====
Linear Regression

Use functions and apps from the Statistics and Machine Learning Toolbox to quickly create and train linear and polynomial regression models.

In this exercise, you will implement linear regression and get to see it work on data.
Files needed for this exercise
ex1.mlx - MATLAB Live Script that steps you through the exercise
ex1data1.txt - Dataset for linear regression with one variable
ex1data2.txt - Dataset for linear regression with multiple variables
submit.m - Submission script that sends your solutions to our servers
*warmUpExercise.m - Simple example function in MATLAB
*plotData.m - Function to display the dataset
*computeCost.m - Function to compute the cost of linear regression
*gradientDescent.m - Function to run gradient descent
**computeCostMulti.m - Cost function for multiple variables
**gradientDescentMulti.m - Gradient descent for multiple variables
**featureNormalize.m - Function to normalize features
**normalEqn.m - Function to compute the normal equations
*indicates files you will need to complete
**indicates optional exercises

Table of Contents
Linear Regression
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. A simple MATLAB function
    1.1 Submitting Solutions
2. Linear regression with one variable
    2.1 Plotting the data
    2.2 Gradient Descent
        2.2.1 Update Equations
        2.2.2 Implementation
        2.2.3 Computing the cost 
        2.2.4 Gradient descent
    2.3 Debugging
    2.4 Visualizing 
Optional Exercises:
3. Linear regression with multiple variables
    3.1 Feature Normalization
        Add the bias term
    3.2 Gradient Descent
        3.2.1 Optional (ungraded) exercise: Selecting learning rates
    3.3 Normal Equations
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 2 
=====
Logistic Regression

Use functions and apps from the Statistics and Machine Learning Toolbox to implement logistic regression.

In this exercise, you will implement logistic regression and apply it to two different datasets.
Files needed for this exercise
ex2.mlx - MATLAB Live Script that steps you through the exercise
ex2data1.txt - Training set for the first half of the exercise
ex2data2.txt - Training set for the second half of the exercise
submit.m - Submission script that sends your solutions to our servers
mapFeature.m - Function to generate polynomial features
plotDecisionBoundary.m - Function to plot classifier's decision boundary
*plotData.m - Function to plot 2D classification data
*sigmoid.m - Sigmoid function
*costFunction.m - Logistic regression cost function
*predict.m - Logistic regression prediction function
*costFunctionReg.m - Regularized logistic regression cost function
*indicates files you will need to complete

Table of Contents
Logistic Regression
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. Logistic Regression
    1.1 Visualizing the data
    1.2 Implementation
        1.2.1 Warmup exercise: sigmoid function
        1.2.2 Cost function and gradient
        Initialize the data
        Compute the gradient
        1.2.3 Learning parameters using fminunc
        1.2.4 Evaluating logistic regression
2. Regularized logistic regression
    2.1 Visualizing the data
    2.2 Feature mapping
    2.3 Cost function and gradient
        2.3.1 Learning parameters using fminunc
    2.4 Plotting the decision boundary
    2.5 Optional (ungraded) exercises
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 3
=====
Multi-class Classification and Neural Networks

Use functions from the Statistics and Machine Learning Toolbox to easily create and train multi-class classification models. Explore an existing neural network created using the Deep Learning Toolbox, then use it to classify digit images. 

In this exercise, you will implement one-vs-all logistic regression and neural networks to recognize hand-written digits.
Files needed for this exercise
ex3.mlx - MATLAB Live Script that steps you through the exercise
ex3data1.mat - Training set of hand-written digits
ex3weights.mat - Initial weights for the neural network exercise
submit.m - Submission script that sends your solutions to our servers
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine (similar to fminunc)
sigmoid.m - Sigmoid function
*lrCostFunction.m - Logistic regression cost function
*oneVsAll.m - Train a one-vs-all multi-class classifier
*predictOneVsAll.m - Predict using a one-vs-all multi-class classifier
*predict.m - Neural network prediction function
*indicates files you will need to complete

Table of Contents
Multi-class Classification and Neural Networks
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. Multi-class Classification
    1.1 Dataset
1.2 Visualizing the data
    1.3 Vectorizing logistic regression
        1.3.1 Vectorizing the cost function
        1.3.2 Vectorizing the gradient
        1.3.3 Vectorizing regularized logistic regression
    1.4 One-vs-all classication
        1.4.1 One-vs-all prediction
2. Neural Networks
    2.1 Model representation
    2.2 Feedforward propagation and prediction
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 4
=====
Neural Networks Learning

Use functions and apps from the Deep Learning Toolbox to create and train a custom neural network.

In this exercise, you will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition.
Files needed for this exercise
ex4.mlx - MATLAB Live Script that steps you through the exercise
ex4data1.mat - Training set of hand-written digits
ex4weights.mat - Neural network parameters for exercise 4
submit.m - Submission script that sends your solutions to our servers
displayData.m - Function to help visualize the dataset
fmincg.m - Function minimization routine (similar to fminunc)
sigmoid.m - Sigmoid function
computeNumericalGradient.m - Numerically compute gradients
checkNNGradients.m - Function to help check your gradients
debugInitializeWeights.m - Function for initializing weights
predict.m - Neural network prediction function
*sigmoidGradient.m - Compute the gradient of the sigmoid function
*randInitializeWeights.m - Randomly initialize weights
*nnCostFunction.m - Neural network cost function
* indicates files you will need to complete

Table of Contents
Neural Networks Learning
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. Neural Networks
    1.1 Visualizing the data
    1.2 Model representation
    1.3 Feedforward and cost function
    1.4 Regularized cost function
2. Backpropagation
    2.1 Sigmoid gradient
    2.2 Random initialization
    2.3 Backpropagation
    2.4 Gradient checking
    2.5 Regularized neural networks
    2.6 Learning parameters using fmincg
3. Visualizing the hidden layer
    3.1 Optional (ungraded) exercise
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 5
=====
Regularized Linear Regression and Bias vs. Variance

Use functions and apps from the Statistics and Machine Learning Toolbox to quickly partition data and automatically cross-validate machine learning models to determine optimal hyperparameter settings.

In this exercise, you will implement regularized linear regression and use it to study models with different bias-variance properties.
Files needed for this exercise
ex5.mlx - MATLAB Live Script that steps you through the exercise
ex5data1.mat - Dataset
submit.m - Submission script that sends your solutions to our servers
featureNormalize.m - Feature normalization function
fmincg.m - Function minimization routine (similar to fminunc)
plotFit.m - Plot a polynomial fit
trainLinearReg.m - Trains linear regression using your cost function
*linearRegCostFunction.m - Regularized linear regression cost function
*learningCurve.m - Generates a learning curve
*polyFeatures.m - Maps data into polynomial feature space
*validationCurve.m - Generates a cross validation curve
*indicates files you will need to complete

Table of Contents
Regularized Linear Regression and Bias vs. Variance
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. Regularized Linear Regression
    1.1 Visualizing the dataset
    1.2 Regularized linear regression cost function
    1.3 Regularized linear regression gradient
    1.4 Fitting linear regression
2. Bias-variance
    2.1 Learning curves
3. Polynomial regression
    3.1 Learning Polynomial Regression
    3.2 Optional (ungraded) exercise: Adjusting the regularization parameter
    3.3 Selecting lambda using a cross validation set
    3.4 Optional (ungraded) exercise: Computing test set error
    3.5 Optional (ungraded) exercise: Plotting learning curves with randomly selected examples
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 6
=====
Support Vector Machines

Use functions and apps from the Statistics and Machine Learning Toolbox to create, train, and cross-validate support vector machine classifiers.

In this exercise, you will be using support vector machines (SVMs) to build a spam classifier. 
Files needed for this exercise
ex6.mlx - MATLAB Live Script that steps you through the exercise
Part 1
ex6data1.mat - Example Dataset 1
ex6data2.mat - Example Dataset 2
ex6data3.mat - Example Dataset 3
svmTrain.m - SVM training function
svmPredict.m - SVM prediction function
plotData.m - Plot 2D data
visualizeBoundaryLinear.m - Plot linear boundary
visualizeBoundary.m - Plot non-linear boundary
linearKernel.m - Linear kernel for SVM
submit.m - Submission script that sends your solutions to our servers
*gaussianKernel.m - Gaussian kernel for SVM
*dataset3Params.m - Parameters to use for Dataset 3
Part 2
spamTrain.mat - Spam training set
spamTest.mat - Spam test set
emailSample1.txt - Sample email 1
emailSample2.txt - Sample email 2
spamSample1.txt - Sample spam 1
spamSample2.txt - Sample spam 2
vocab.txt - Vocabulary list
getVocabList.m - Load vocabulary list
porterStemmer.m - Stemming function
readFile.m - Reads a file into a character string
submit.m - Submission script that sends your solutions to our servers
*processEmail.m - Email preprocessing
*emailFeatures.m - Feature extraction from emails
*indicates files you will need to complete

Table of Contents
Support Vector Machines
    Files needed for this exercise
        Part 1
        Part 2
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. Support Vector Machines
    1.1 Example dataset 1
    1.2 SVM with gaussian kernels
        1.2.1 Gaussian kernel
        1.2.2 Example dataset 2
        1.2.3 Example dataset 3
2. Spam Classification
    2.1 Preprocessing emails
        2.1.1 Vocabulary list
    2.2 Extracting features from emails
    2.3 Training SVM for spam classification
    2.4 Top predictors for spam
    2.5 Optional (ungraded) exercise: Try your own emails
    2.6 Optional (ungraded) exercise: Build your own dataset
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 7
=====
K-Means Clustering and Principal Component Analysis

Use functions from the Statistics and Machine Learning Toolbox to cluster data and determine the optimal number of clusters. Then learn how to compress data using PCA and automatically include data compression when using the MATLAB machine learning apps.

In this exercise, you will implement the K-means clustering algorithm and apply it to compress an image. In the second part, you will use principal component analysis to find a low-dimensional representation of face images.
Files needed for this exercise
ex7.mlx - MATLAB Live Script that steps you through the exercise
ex7data1.mat - Example Dataset for PCA
ex7data2.mat - Example Dataset for K-means
ex7faces.mat - Faces Dataset
bird small.png - Example Image
displayData.m - Displays 2D data stored in a matrix
drawLine.m - Draws a line over an exsiting figure
plotDataPoints.m - Initialization for K-means centroids
plotProgresskMeans.m - Plots each step of K-means as it proceeds
runkMeans.m - Runs the K-means algorithm
submit.m - Submission script that sends your solutions to our servers
*pca.m - Perform principal component analysis
*projectData.m - Projects a data set into a lower dimensional space
*recoverData.m - Recovers the original data from the projection
*findClosestCentroids.m - Find closest centroids (used in K-means)
*computeCentroids.m - Compute centroid means (used in K-means)
*kMeansInitCentroids.m - Initialization for K-means centroids
*indicates files you will need to complete

Table of Contents
K-Means Clustering and Principal Component Analysis
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. K-Means Clustering
    1.1 Implementing K-means
        1.1.1 Finding closest centroids
        1.1.2 Computing centroid means
    1.2 K-means on example dataset
    1.3 Random initialization
    1.4 Image compression with K-means
        1.4.1 K-means on pixels
    1.5 Optional (ungraded) exercise: Use your own image
2. Principal Component Analysis
    2.1 Example dataset
    2.2 Implementing PCA
    2.3 Dimensionality reduction with PCA
        2.3.1 Projecting the data onto the principal components
        2.3.2 Reconstructing an approximation of the data
        2.3.3 Visualizing the projections
    2.4 Face image dataset
        2.4.1 PCA on faces
    2.5 Optional (ungraded) exercise: PCA for visualization
Submission and Grading
----------------------------------------------------------------------------

Machine Learning: Programming Exercise 8
=====
Anomaly Detection and Recommender Systems

Use MATLAB functionality for working with big data to analyze movie ratings data and implement recommender systems using sparse arrays. 

In this exercise, you will implement the anomaly detection algorithm and apply it to detect failing servers on a network. In the second part, you will use collaborative filtering to build a recommender system for movies.
Files needed for this exercise
ex8.mlx - MATLAB Live Script that steps you through the exercise
ex8data1.mat - First example Dataset for anomaly detection
ex8data2.mat - Second example Dataset for anomaly detection
ex8_movies.mat - Movie Review Dataset
ex8_movieParams.mat - Parameters provided for debugging
multivariateGaussian.m - Computes the probability density function for a Gaussian distribution
visualizeFit.m - 2D plot of a Gaussian distribution and a dataset
checkCostFunction.m - Gradient checking for collaborative filtering
computeNumericalGradient.m - Numerically compute gradients
fmincg.m - Function minimization routine (similar to fminunc)
loadMovieList.m - Loads the list of movies into a cell-array
movie_ids.txt - List of movies
normalizeRatings.m - Mean normalization for collaborative filtering
submit.m - Submission script that sends your solutions to our servers
*estimateGaussian.m - Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix
*selectThreshold.m - Find a threshold for anomaly detection
*cofiCostFunc.m - Implement the cost function for collaborative filtering
* indicates files you will need to complete

Table of Contents
Anomaly Detection and Recommender Systems
    Files needed for this exercise
        Clear existing variables and confirm that your Current Folder is set correctly
    Before you begin
1. Anomaly Detection
    1.1 Gaussian distribution
    1.2 Estimating parameters for a Gaussian
    1.3 Selecting the threshold, 
    1.4 High dimensional dataset
2. Recommender Systems
    2.1 Movie ratings dataset
    2.2 Collaborative filtering learning algorithm
        2.2.1 Collaborative filtering cost function
        2.2.2 Collaborative filtering gradient
        2.2.3 Regularized cost function
        2.2.4 Regularized gradient
    2.3 Learning movie recommendations
        2.3.1 Recommendations
Submission and Grading

