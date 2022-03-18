# Machine Learning: Programming Exercise 8
# Anomaly Detection and Recommender Systems

In this exercise, you will implement the anomaly detection algorithm and apply it to detect failing servers on a network. In the second part, you will use collaborative filtering to build a recommender system for movies.

## Files needed for this exercise

   -  `ex8.mlx` - MATLAB Live Script that steps you through the exercise 
   -  `ex8data1.mat` - First example Dataset for anomaly detection 
   -  `ex8data2.mat` - Second example Dataset for anomaly detection 
   -  `ex8_movies.mat` - Movie Review Dataset 
   -  `ex8_movieParams.mat` - Parameters provided for debugging 
   -  `multivariateGaussian.m` - Computes the probability density function for a Gaussian distribution 
   -  `visualizeFit.m` - 2D plot of a Gaussian distribution and a dataset 
   -  `checkCostFunction.m` - Gradient checking for collaborative filtering 
   -  `computeNumericalGradient.m` - Numerically compute gradients 
   -  `fmincg.m` - Function minimization routine (similar to `fminunc`) 
   -  `loadMovieList.m` - Loads the list of movies into a cell-array 
   -  movie_ids.txt - List of movies 
   -  `normalizeRatings.m` - Mean normalization for collaborative filtering 
   -  `submit.m` - Submission script that sends your solutions to our servers 
   -  *`estimateGaussian.m` - Estimate the parameters of a Gaussian distribution with a diagonal covariance matrix 
   -  *`selectThreshold.m` - Find a threshold for anomaly detection 
   -  *`cofiCostFunc.m` - Implement the cost function for collaborative filtering 

**** indicates files you will need to complete***

### Clear existing variables and confirm that your Current Folder is set correctly

Click into this section, then click the 'Run Section' button above. This will execute the `clear` command to clear existing variables and the `dir` command to list the files in your Current Folder. The output should contain all of the files listed above and the 'lib' folder. If it does not, right-click the 'ex8' folder and select 'Open' before proceding or see the instructions in `README.mlx` for more details.

```matlab:Code
clear
dir
```

```text:Output
.                           ex8.mlx                     ex8data2.mat                selectThreshold.m           
..                          ex8_companion1.mlx          fmincg.m                    submit.m                    
checkCostFunction.m         ex8_companion2.mlx          loadMovieList.m             visualizeFit.m              
cofiCostFunc.m              ex8_movieParams.mat         movie_ids.txt               
computeNumericalGradient.m  ex8_movies.mat              multivariateGaussian.m      
estimateGaussian.m          ex8data1.mat                normalizeRatings.m          
```

## Before you begin

The workflow for completing and submitting the programming exercises in MATLAB Online differs from the original course instructions. Before beginning this exercise, make sure you have read through the instructions in `README.mlx` which is included with the programming exercise files. `README` also contains solutions to the many common issues you may encounter while completing and submitting the exercises in MATLAB Online. Make sure you are following instructions in `README` and have checked for an existing solution before seeking help on the discussion forums.

# 1. Anomaly Detection

In this exercise, you will implement an anomaly detection algorithm to detect anomalous behavior in server computers. The features measure the throughput (mb/s) and latency (ms) of response of each server. While your servers were operating, you collected <img src="https://latex.codecogs.com/gif.latex?\inline&space;m=307"/> examples of how they were behaving, and thus have an unlabeled dataset <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;x^{(1)}&space;,\ldots,x^{(m)}&space;\rbrace"/>. You suspect that the vast majority of these examples are 'normal' (non-anomalous) examples of the servers operating normally, but there might also be some examples of servers acting anomalously within this dataset.

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_0.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_0.png'
)

    You will use a Gaussian model to detect anomalous examples in your dataset. You will first start on a 2D dataset that will allow you to visualize what the algorithm is doing. On that dataset you will fit a Gaussian distribution and then find values that have very low probability and hence can be considered anomalies. After that, you will apply the anomaly detection algorithm to a larger dataset with many dimensions. The code below will visualize the dataset as shown in Figure 1.

```matlab:Code
% The following command loads the dataset. You should now have the variables X, Xval, yval in your environment
load('ex8data1.mat');

% Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
```

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_0.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_0.png'
)

## 1.1 Gaussian distribution

To perform anomaly detection, you will first need to fit a model to the data's distribution. Given a training set <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;x^{(1)}&space;,\ldots,x^{(m)}&space;\rbrace"/> (where <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}&space;\in&space;{\mathbb{R}}^n"/>), you want to estimate the Gaussian distribution for each of the features <img src="https://latex.codecogs.com/gif.latex?\inline&space;x_i"/>. For each feature <img src="https://latex.codecogs.com/gif.latex?\inline&space;i=1\ldotsn"/> you need to find parameters <img src="https://latex.codecogs.com/gif.latex?\inline&space;\mu_i"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma_i^2"/> that fit the data in the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th dimension <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;x_i^{(1)}&space;,\ldots,x_i^{(m)}&space;\rbrace"/>(the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th dimension of each example).

    The Gaussian distribution is given by

<img src="https://latex.codecogs.com/gif.latex?p(x;\mu&space;,\sigma^2&space;)=\frac{1}{\sqrt{2\pi&space;\sigma^2&space;}}e^{-\frac{(x-\mu&space;)^2&space;}{2\sigma^2&space;}}"/>

where <img src="https://latex.codecogs.com/gif.latex?\inline&space;\mu"/> is the mean and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma^2"/> controls the variance.

## 1.2 Estimating parameters for a Gaussian

You can estimate the parameters, <img src="https://latex.codecogs.com/gif.latex?\inline&space;(\mu_i&space;,\sigma_i^2&space;)"/>, of the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th feature by using the following equations. To estimate the mean, you will use:

> <img src="https://latex.codecogs.com/gif.latex?\inline&space;\mu_i&space;=\frac{1}{m}\sum_{j=1}^m&space;x^{(j)}"/>,

and for the variance you will use:

> <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma_i^2&space;=\frac{1}{m}\sum_{j=1}^m&space;(x^{(j)}&space;-\mu_i&space;)^2"/>.

Your task is to complete the code in `estimateGaussian.m`. This function takes as input the data matrix `X` and should output an <img src="https://latex.codecogs.com/gif.latex?\inline&space;n"/>-dimension vector `mu` that holds the mean of all the <img src="https://latex.codecogs.com/gif.latex?\inline&space;n"/> features and another <img src="https://latex.codecogs.com/gif.latex?\inline&space;n"/>-dimension vector `sigma2` that holds the variances of all the features. You can implement this using a `for `loop over every feature and every training example (though a vectorized implementation might be more efficient; feel free to use a vectorized implementation if you prefer). Note that in MATLAB, the `var` function will (by default) use <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{1}{m-1}"/>, instead of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{1}{m}"/> when computing <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma_i^2"/>.

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_1.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_1.png'
)

    Once you have completed the code in `estimateGaussian.m`, the code below will visualize the contours of the fitted Gaussian distribution. You should get a plot similar to Figure 2. From your plot, you can see that most of the examples are in the region with the highest probability, while the anomalous examples are in the regions with lower probabilities.

```matlab:Code
%  Estimate mu and sigma2
[mu, sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
```

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_1.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_1.png'
)

> *You should now submit your solutions.  Enter ***`submit`*** at the command prompt, then and enter or confirm your login and token when prompted.*

## 1.3 Selecting the threshold, <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/>

Now that you have estimated the Gaussian parameters, you can investigate which examples have a very high probability given this distribution and which examples have a very low probability. The low probability examples are more likely to be the anomalies in our dataset. One way to determine which examples are anomalies is to select a threshold based on a cross validation set. In this part of the exercise, you will implement an algorithm to select the threshold <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/> using the <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1"/> score on a cross validation set.

    You should now complete the code in `selectThreshold.m`. For this, we will use a cross validation set <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lbrace&space;(x_{cv}^{(1)}&space;,y_{cv}^{(1)}&space;),\ldots,(x_{cv}^{(m)}&space;,y_{cv}^{(m)}&space;)\rbrace"/>, where the label <img src="https://latex.codecogs.com/gif.latex?\inline&space;y=1"/> corresponds to an anomalous example, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;y=0"/> corresponds to a normal example. For each cross validation example, we will compute <img src="https://latex.codecogs.com/gif.latex?\inline&space;p(x_{cv}^{(i)}&space;)"/>. The vector of all of these probabilities <img src="https://latex.codecogs.com/gif.latex?\inline&space;p(x_{cv}^{(1)}&space;),\ldots,p(x_{cv}^{(m)}&space;)"/> is passed to `selectThreshold.m` in the vector `pval`. The corresponding labels <img src="https://latex.codecogs.com/gif.latex?\inline&space;y_{cv}^{(1)}&space;,\ldots,y_{cv}^{m)}"/> is passed to the same function in the vector `yval`.

    The function `selectThreshold.m` should return two values; the first is the selected threshold <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/>. If an example <img src="https://latex.codecogs.com/gif.latex?\inline&space;x"/> has a low probability <img src="https://latex.codecogs.com/gif.latex?\inline&space;p(x)<\epsilon"/>, then it is considered to be an anomaly. The function should also return the <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1"/> score, which tells you how well you're doing on finding the ground truth anomalies given a certain threshold. For many different values of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/>, you will compute the resulting <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1"/> score by computing how many examples the current threshold classifies correctly and incorrectly.

    The <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1"/> score is computed using precision (*prec*) and recall (*rec*):

> <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1&space;=\frac{2(prec)(rec)}{prec+rec}"/>,

You compute precision and recall by:

<img src="https://latex.codecogs.com/gif.latex?prec=\frac{tp}{tp+fp}"/>

<img src="https://latex.codecogs.com/gif.latex?rec=\frac{tp}{tp+fn}"/>

where

   -  <img src="https://latex.codecogs.com/gif.latex?\inline&space;tp"/> is the number of true positives: the ground truth label says it's an anomaly and our algorithm correctly classified it as an anomaly. 
   -  <img src="https://latex.codecogs.com/gif.latex?\inline&space;fp"/> is the number of false positives: the ground truth label says it's not an anomaly, but our algorithm incorrectly classified it as an anomaly. 
   -  <img src="https://latex.codecogs.com/gif.latex?\inline&space;fn"/> is the number of false negatives: the ground truth label says it's an anomaly, but our algorithm incorrectly classified it as not being anomalous. 

In the provided code `selectThreshold.m`, there is already a loop that will try many different values of <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/> and select the "best" based <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/> on the <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1"/> score. You should now complete the code in `selectThreshold.m`. You can implement the computation of the <img src="https://latex.codecogs.com/gif.latex?\inline&space;F_1"/> score using a `for` loop over all the cross validation examples (to compute the values <img src="https://latex.codecogs.com/gif.latex?\inline&space;tp,\;fp,\textrm{and}\;fn"/>. You should see a value for epsilon of about `8.99e-05`.

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_2.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_2.png'
)

**Implementation Note**: In order to compute <img src="https://latex.codecogs.com/gif.latex?\inline&space;tp,\;fp,\textrm{and}\;fn"/>, you may be able to use a vectorized implementation rather than loop over all the examples. This can be implemented by MATLAB's equality test between a vector and a single number. If you have several binary values in an <img src="https://latex.codecogs.com/gif.latex?\inline&space;n"/>-dimensional binary vector <img src="https://latex.codecogs.com/gif.latex?\inline&space;v\in&space;\lbrace&space;0,\;1\rbrace^n"/>, you can find out how many values in this vector are 0 by using: `sum(v == 0)`. You can also apply a logical `and` operator to such binary vectors. For instance, let `cvPredictions` be a binary vector of the size of your cross validation set, where the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th element is 1 if your algorithm considers <img src="https://latex.codecogs.com/gif.latex?\inline&space;x_{cv}^{(i)}"/> an anomaly, and 0 otherwise. You can then, for example, compute the number of false positives using:` fp = sum((cvPredictions == 1) \& (yval == 0))`.

    Once you have completed the code in `selectThreshold.m`, the code below will detect and circle the anomalies in the plot (Figure 3).

```matlab:Code
pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
```

```text:Output
Best epsilon found using cross-validation: 8.990853e-05
```

```matlab:Code
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
```

```text:Output
Best F1 on Cross Validation Set:  0.875000
```

```matlab:Code

%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');
%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off
```

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_2.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_2.png'
)

> *You should now submit your solutions.  Enter ***`submit`*** at the command prompt, then and enter or confirm your login and token when prompted.*

## 1.4 High dimensional dataset

The code in this section will run the anomaly detection algorithm you implemented on a more realistic and much harder dataset. In this dataset, each example is described by 11 features, capturing many more properties of your compute servers. The code below will use your code to estimate the Gaussian parameters (<img src="https://latex.codecogs.com/gif.latex?\inline&space;\mu_i"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\sigma_i^2"/>), evaluate the probabilities for both the training data `X` from which you estimated the Gaussian parameters, and do so for the the cross-validation set `Xval`. Finally, it will use `selectThreshold` to find the best threshold <img src="https://latex.codecogs.com/gif.latex?\inline&space;\epsilon"/>. You should see a value `epsilon` of about `1.38e-18`, and 117 anomalies found.

```matlab:Code
%  Loads the second dataset. You should now have the variables X, Xval, yval in your environment
load('ex8data2.mat');

%  Apply the same steps to the larger dataset
[mu, sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon, F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
```

```text:Output
Best epsilon found using cross-validation: 1.377229e-18
```

```matlab:Code
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
```

```text:Output
Best F1 on Cross Validation Set:  0.615385
```

```matlab:Code
fprintf('# Outliers found: %d\n', sum(p < epsilon));
```

```text:Output
# Outliers found: 117
```

# 2. Recommender Systems

In this part of the exercise, you will implement the collaborative filtering learning algorithm and apply it to a dataset of movie ratings*. This dataset consists of ratings on a scale of 1 to 5. The dataset has <img src="https://latex.codecogs.com/gif.latex?\inline&space;n_u&space;=943"/> users, and <img src="https://latex.codecogs.com/gif.latex?\inline&space;n_m&space;=1682"/> movies. In the next parts of this exercise, you will implement the function `cofiCostFunc.m` that computes the collaborative fitlering objective function and gradient. After implementing the cost function and gradient, you will use `fmincg.m` to learn the parameters for collaborative filtering.

*[MovieLens 100k Dataset](https://grouplens.org/datasets/movielens/) from GroupLens Research.

## 2.1 Movie ratings dataset

The code in this section will load the dataset `ex8_movies.mat`, providing the variables `Y` and `R` in your MATLAB environment. The matrix `Y` (a `num_movies` <img src="https://latex.codecogs.com/gif.latex?\inline&space;\times"/> `num_users` matrix) stores the ratings <img src="https://latex.codecogs.com/gif.latex?\inline&space;y^{(i,j)}"/> (from 1 to 5). The matrix `R` is an binary-valued indicator matrix, where` R(i,j) = 1` if user `j` gave a rating to movie `i`, and `R(i,j) = 0` otherwise. The objective of collaborative filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries with `R(i,j) = 0`. This will allow us to recommend the movies with the highest predicted ratings to the user.

    To help you understand the matrix `Y`, the code below will compute the average movie rating for the first movie (Toy Story) and output the average rating to the screen. Throughout this part of the exercise, you will also be working with the matrices, `X` and `Theta`:

> `X = `$\left\lbrack \begin{array}{c}
-{\;\left(x^{\left(1\right)} \right)}^T -\\
-{\;\left(x^{\left(2\right)} \right)}^T -\\
\vdots \\
-{\;\left(x^{\left(n_m \right)} \right)}^T -
\end{array}\right\rbrack<img src="https://latex.codecogs.com/gif.latex?\inline&space;,&space;&space;&space;&space;`Theta&space;=&space;`"/>\left\lbrack \begin{array}{c}
-{\;\left(\theta^{\left(1\right)} \right)}^T -\\
-{\;\left(\theta^{\left(2\right)} \right)}^T -\\
\vdots \\
-{\;\left(\theta^{\left(n_u \right)} \right)}^T -
\end{array}\right\rbrack$.

  

```matlab:Code
% Load data
load('ex8_movies.mat');
```

   -  `Y` is a 1682 x 943 matrix, containing ratings (1 - 5) of 1682 movies on 943 users 
   -  `R` is a 1682 x 943 matrix, where `R(i,j)` = 1 if and only if user `j` gave a rating to movie `i` 

```matlab:Code
% From the matrix, we can compute statistics like average rating.
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', mean(Y(1, R(1, :))));
```

```text:Output
Average rating for movie 1 (Toy Story): 3.878319 / 5
```

```matlab:Code
%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
ylabel('Movies');
xlabel('Users');
```

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_3.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/figure_3.png'
)

The <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th row of `X` corresponds to the feature vector <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}"/> for the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th movie, and the <img src="https://latex.codecogs.com/gif.latex?\inline&space;j"/>-th row of `Theta` corresponds to one parameter vector <img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta^{(j)}"/>, for the <img src="https://latex.codecogs.com/gif.latex?\inline&space;j"/>-th user. Both <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta^{(j)}"/> are <img src="https://latex.codecogs.com/gif.latex?\inline&space;n"/>-dimensional vectors. For the purposes of this exercise, you will use <img src="https://latex.codecogs.com/gif.latex?\inline&space;n=100"/>, and therefore, <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}&space;\in&space;{\mathbb{R}}^{100}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta^{(j)}&space;\in&space;{\mathbb{R}}^{100}"/>. Correspondingly, `X` is a <img src="https://latex.codecogs.com/gif.latex?\inline&space;n_m&space;\times&space;100"/> matrix and `Theta` is a <img src="https://latex.codecogs.com/gif.latex?\inline&space;n_u&space;\times&space;100"/> matrix.

## 2.2 Collaborative filtering learning algorithm

Now, you will start implementing the collaborative filtering learning algorithm. You will start by implementing the cost function (without regularization). The collaborative filtering algorithm in the setting of movie recommendations considers a set of <img src="https://latex.codecogs.com/gif.latex?\inline&space;n"/>-dimensional parameter vectors <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(1)}&space;,\ldots,x^{(n_m&space;)}"/> and <img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta^{(1)}&space;,\ldots,\theta^{(n_u&space;)}"/>, where the model predicts the rating for movie <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/> by user <img src="https://latex.codecogs.com/gif.latex?\inline&space;j"/> as <img src="https://latex.codecogs.com/gif.latex?\inline&space;y^{(i,j)}&space;=(\theta^{(j)}&space;)^T&space;x^{(i)}"/>. Given a dataset that consists of a set of ratings produced by some users on some movies, you wish to learn the parameter vectors <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(1)}&space;,\ldots,x^{(n_m&space;)}&space;,\;\theta^{(1)}&space;,\ldots,\theta^{(n_u&space;)}"/> that produce the best fit (minimizes the squared error).

    You will complete the code in `cofiCostFunc.m` to compute the cost function and gradient for collaborative filtering. Note that the parameters to the function (i.e., the values that you are trying to learn) are `X` and `Theta`. In order to use an off-the-shelf minimizer such as `fmincg`, the cost function has been set up to unroll the parameters into a single vector `params`. You had previously used the same vector unrolling method in the neural networks programming exercise.

### 2.2.1 Collaborative filtering cost function

The collaborative filtering cost function (without regularization) is given by

<img src="https://latex.codecogs.com/gif.latex?J\left(x^{(i)}&space;,\ldots,x^{(n_m&space;)}&space;,\;\theta^{(1)}&space;,\ldots,\theta^{(n_u&space;)}&space;\right)=\frac{1}{2}\sum_{(i,j):r(i,j)=1}&space;{\left((\theta^{(j)}&space;)^T&space;x^{(i)}&space;-y^{(i,j)}&space;\right)}^2"/>

You should now modify `cofiCostFunc.m` to return this cost in the variable `J`. Note that you should be accumulating the cost for user <img src="https://latex.codecogs.com/gif.latex?\inline&space;j"/> and movie <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/> only if <img src="https://latex.codecogs.com/gif.latex?\inline&space;R(i,j)=1"/>. After you have completed the function, the code below will run your cost function. You should expect to see an output of `22.22`.

  

**Implementation Note:** We strongly encourage you to use a vectorized implementation to compute `J`, since it will later by called many times by the optimization package `fmincg`. As usual, it might be easiest to first write a non-vectorized implementation (to make sure you have the right answer), and the modify it to become a vectorized implementation (checking that the vectorization steps don't change your algorithm's output). To come up with a vectorized implementation, the following tip might be helpful: You can use the `R` matrix to set selected entries to 0. For example, \texttt{R.*M} will do an element-wise multiplication between `M` and `R`; since `R` only has elements with values either 0 or 1, this has the effect of setting the elements of `M` to 0 only when the corresponding value in `R` is 0. Hence, \texttt{sum(sum(R.*M))} is the sum of all the elements of `M` for which the corresponding element in `R` equals 1.

```matlab:Code
%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc([X(:); Theta(:)],  Y, R, num_users, num_movies,num_features, 0);
fprintf('Cost at loaded parameters: %f ',J);
```

```text:Output
Cost at loaded parameters: 22.224604 
```

> *You should now submit your solutions.  Enter ***`submit`*** at the command prompt, then and enter or confirm your login and token when prompted.*

### 2.2.2 Collaborative filtering gradient

Now, you should implement the gradient (without regularization). Specically, you should complete the code in `cofiCostFunc.`m to return the variables `X_grad` and `Theta_gra`d. Note that `X_grad` should be a matrix of the same size as `X` and similarly, `Theta_grad` is a matrix of the same size as `Theta`. The gradients of the cost function is given by:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J}{\partial&space;x_k^{(i)}&space;}=\sum_{j:r(i,j)=1}&space;\left((\theta^{(j)}&space;)^T&space;x^{(i)}&space;-y^{(i,j)}&space;\right)\theta_k^{(j)}"/>

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J}{\partial&space;\theta_k^{(j)}&space;}=\sum_{i:r(i,j)=1}&space;\left((\theta^{(j)}&space;)^T&space;x^{(i)}&space;-y^{(i,j)}&space;\right)x_k^{(i)}"/>

Note that the function returns the gradient for both sets of variables by unrolling them into a single vector. After you have completed the code to compute the gradients, the code below will run a gradient check (`checkCostFunction`) to numerically check the implementation of your gradients. (This is similar to the numerical check that you used in the neural networks exercise.) If your implementation is correct, you should find that the analytical and numerical gradients match up closely.

  

**Implementation Note**: You can get full credit for this assignment without using a vectorized implementation, but your code will run much more slowly (a small number of hours), and so we recommend that you try to vectorize your implementation. To get started, you can implement the gradient with a `for` loop over movies (for computing <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;J}{\partial&space;x_k^{(i)}&space;}"/>) and a `for` loop over users (for computing <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;J}{\partial&space;\theta_k^{(j)}&space;}"/>). When you first implement the gradient, you might start with an unvectorized version, by implementing another inner `for` loop that computes each element in the summation. After you have completed the gradient computation this way, you should try to vectorize your implementation (vectorize the inner `for` loops), so that you're left with only two `for` loops (one for looping over movies to compute <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;J}{\partial&space;x_k^{(i)}&space;}"/> for each movie, and one for looping over users to compute <img src="https://latex.codecogs.com/gif.latex?\inline&space;\frac{\partial&space;J}{\partial&space;\theta_k^{(j)}&space;}"/> for each user).

    To perform the vectorization, you might find this helpful: You should come up with a way to compute all the derivatives associated with <img src="https://latex.codecogs.com/gif.latex?\inline&space;x_1^{(i)}&space;,x_2^{(i)}&space;,\ldots,x_n^{(i)}"/>, (i.e., the derivative terms associated with the feature vector <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}"/>) at the same time. Let us define the derivatives for the feature vector of the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th movie as:

<img src="https://latex.codecogs.com/gif.latex?{\left({\mathrm{X}}_{\textrm{grad}}&space;\left(\mathrm{i},:\right)\right)}^T&space;=\left\lbrack&space;\begin{array}{c}&space;\frac{\partial&space;J}{\partial&space;x_1^{\left(i\right)}&space;}\\&space;\frac{\partial&space;J}{\partial&space;x_2^{\left(i\right)}&space;}\\&space;\vdots&space;\\&space;\frac{\partial&space;J}{\partial&space;x_n^{\left(i\right)}&space;}&space;\end{array}\right\rbrack&space;=\sum_{j:r\left(i,j\right)=1}&space;\left({\left(\theta^{\left(j\right)}&space;\right)}^T&space;x^{\left(i\right)}&space;-y^{\left(i,j\right)}&space;\right)\;\theta^{\left(j\right)}"/>

To vectorize the above expression, you can start by indexing into `Theta` and `Y` to select only the elements of interest (that is, those with <img src="https://latex.codecogs.com/gif.latex?\inline&space;r(i,j)=1"/>). Intuitively, when you consider the features for the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th movie, you only need to be concerned about the users who had given ratings to the movie, and this allows you to remove all the other users from `Theta` and `Y`.

    Concretely, you can set `idx = find(R(i,:)==1)` to be a list of all the users that have rated movie <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>. This will allow you to create the temporary matrices `Theta_temp = Theta(idx,:)` and `Y_temp = Y(i,idx)` that index into `Theta` and `Y` to give you only the set of users which have rated the <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/>-th movie. This will allow you to write the derivatives as:

> <img src="https://latex.codecogs.com/gif.latex?\inline&space;{\mathrm{X}}_{\textrm{grad}}&space;\left(i,:\right)=\left(\mathrm{X}\left(i,:\right)*{\Theta&space;\;}_{\textrm{temp}}^T&space;-{\mathrm{Y}}_{\textrm{temp}}&space;\right)*{\Theta&space;\;}_{\textrm{temp}}"/>.

(Note: The vectorized computation above returns a row-vector instead.) After you have vectorized the computations of the derivatives with respect to <img src="https://latex.codecogs.com/gif.latex?\inline&space;x^{(i)}"/>, you should use a similar method to vectorize the derivatives with respect to <img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta^{(j)}"/> as well.

```matlab:Code
%  Check gradients by running checkNNGradients
checkCostFunction;
```

```text:Output
    5.5335    5.5335
    3.6186    3.6186
    5.4422    5.4422
   -1.7312   -1.7312
    4.1196    4.1196
   -1.4833   -1.4833
   -6.0734   -6.0734
    2.3490    2.3490
    7.6341    7.6341
    1.8651    1.8651
    4.1192    4.1192
   -1.5834   -1.5834
    1.2828    1.2828
   -6.1573   -6.1573
    1.6628    1.6628
    1.1686    1.1686
    5.5630    5.5630
    0.3050    0.3050
    4.6442    4.6442
   -1.6691   -1.6691
   -2.1505   -2.1505
   -3.6832   -3.6832
    3.4067    3.4067
   -4.0743   -4.0743
    0.5567    0.5567
   -2.1056   -2.1056
    0.9168    0.9168

The above two columns you get should be very similar.
(Left-Your Numerical Gradient, Right-Analytical Gradient)

If your cost function implementation is correct, then 
the relative difference will be small (less than 1e-9). 

Relative Difference: 1.50285e-12
```

> *You should now submit your solutions.  Enter ***`submit`*** at the command prompt, then and enter or confirm your login and token when prompted.*

### 2.2.3 Regularized cost function

The cost function for collaborative filtering with regularization is given by

<img src="https://latex.codecogs.com/gif.latex?\begin{array}{l}&space;J\left(x^{(i)}&space;,\ldots,x^{(n_m&space;)}&space;,\;\theta^{(1)}&space;,\ldots,\theta^{(n_u&space;)}&space;\right)=\\&space;~~~~~~~~\frac{1}{2}\sum_{(i,j):r(i,j)=1}&space;{\left((\theta^{(j)}&space;)^T&space;x^{(i)}&space;-y^{(i,j)}&space;\right)}^2&space;+\left(\frac{\lambda&space;}{2}\sum_{j=1}^{n_u&space;}&space;\sum_{k=1}^n&space;(\theta_k^{(j)}&space;)^2&space;\right)+\left(\frac{\lambda&space;}{2}\sum_{i=1}^{n_m&space;}&space;\sum_{k=1}^n&space;(x_k^{(i)}&space;)^2&space;\right)&space;\end{array}"/>

You should now add regularization to your original computations of the cost function, <img src="https://latex.codecogs.com/gif.latex?\inline&space;J"/>. After you are done, the code below will run your regularized cost function, and you should expect to see a cost of about `31.34`.

```matlab:Code
%  Evaluate cost function
J = cofiCostFunc([X(:); Theta(:)], Y, R, num_users, num_movies, num_features, 1.5);      
fprintf('Cost at loaded parameters (lambda = 1.5): %f',J);
```

```text:Output
Cost at loaded parameters (lambda = 1.5): 31.344056
```

> *You should now submit your solutions.  Enter ***`submit`*** at the command prompt, then and enter or confirm your login and token when prompted.*

### 2.2.4 Regularized gradient

Now that you have implemented the regularized cost function, you should proceed to implement regularization for the gradient. You should add to your implementation in `cofiCostFunc.m` to return the regularized gradient by adding the contributions from the regularization terms. Note that the gradients for the regularized cost function is given by:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J}{\partial&space;x_k^{(i)}&space;}=\sum_{j:r(i,j)=1}&space;\left((\theta^{(j)}&space;)^T&space;x^{(i)}&space;-y^{(i,j)}&space;\right)\theta_k^{(j)}&space;+\lambda&space;x_k^{(i)}"/>

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;J}{\partial&space;\theta_k^{(j)}&space;}=\sum_{i:r(i,j)=1}&space;\left((\theta^{(j)}&space;)^T&space;x^{(i)}&space;-y^{(i,j)}&space;\right)x_k^{(i)}&space;+\lambda&space;\theta_k^{(j)}"/>

This means that you just need to add <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda&space;x^{(i)}"/> to the `X_grad(i,:)` variable described earlier, and add <img src="https://latex.codecogs.com/gif.latex?\inline&space;\lambda&space;\theta^{(j)}"/> to the `Theta_grad(j,:) `variable described earlier. After you have completed the code to compute the gradients, the code below will run another gradient check (`checkCostFunction`) to numerically check the implementation of your gradients.

```matlab:Code
%  Check gradients by running checkNNGradients
checkCostFunction(1.5);
```

```text:Output
    2.2223    2.2223
    0.7968    0.7968
   -3.2924   -3.2924
   -0.7029   -0.7029
   -4.2016   -4.2016
    3.5969    3.5969
    0.8859    0.8859
    1.0523    1.0523
   -7.8499   -7.8499
    0.3904    0.3904
   -0.1347   -0.1347
   -2.3656   -2.3656
    2.1066    2.1066
    1.6703    1.6703
    0.8519    0.8519
   -1.0380   -1.0380
    2.6537    2.6537
    0.8114    0.8114
   -0.8604   -0.8604
   -0.5884   -0.5884
   -0.7108   -0.7108
   -4.0652   -4.0652
    0.2494    0.2494
   -4.3484   -4.3484
   -3.6167   -3.6167
   -4.1277   -4.1277
   -3.2439   -3.2439

The above two columns you get should be very similar.
(Left-Your Numerical Gradient, Right-Analytical Gradient)

If your cost function implementation is correct, then 
the relative difference will be small (less than 1e-9). 

Relative Difference: 2.90579e-12
```

> *You should now submit your solutions.  Enter ***`submit`*** at the command prompt, then and enter or confirm your login and token when prompted.*

## 2.3 Learning movie recommendations

After you have finished implementing the collaborative ltering cost function and gradient, you can now start training your algorithm to make movie recommendations for yourself. In the code below, you can enter your own movie preferences, so that later when the algorithm runs, you can get your own movie recommendations! We have filled out some values according to our own preferences, but you should change this according to your own tastes. The list of all movies and their number in the dataset can be found listed in the file movie idx.txt.

```matlab:Code
% Load movvie list
movieList = loadMovieList();

% Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;
% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2; 

% We have selected a few movies we liked / did not like and the ratings we gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

fprintf('\n\nNew user ratings:\n');
```

```text:Output
New user ratings:
```

```matlab:Code
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end
```

```text:Output
Rated 4 for Toy Story (1995)
Rated 3 for Twelve Monkeys (1995)
Rated 5 for Usual Suspects, The (1995)
Rated 4 for Outbreak (1995)
Rated 5 for Shawshank Redemption, The (1994)
Rated 3 for While You Were Sleeping (1995)
Rated 5 for Forrest Gump (1994)
Rated 2 for Silence of the Lambs, The (1991)
Rated 4 for Alien (1979)
Rated 5 for Die Hard 2 (1990)
Rated 5 for Sphere (1998)
```

### 2.3.1 Recommendations

After the additional ratings have been added to the dataset, the code below will proceed to train the collaborative filtering model. This will learn the parameters `X` and `Theta`. 

```matlab:Code
%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 943 users
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj','on','MaxIter',100);

% Set Regularization
lambda = 10;
theta = fmincg(@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, num_features,lambda)), initial_parameters, options);
```

```text:Output
Iteration     1 | Cost: 3.108511e+05
Iteration     2 | Cost: 1.475959e+05
Iteration     3 | Cost: 1.000321e+05
Iteration     4 | Cost: 7.707565e+04
Iteration     5 | Cost: 6.153638e+04
Iteration     6 | Cost: 5.719300e+04
Iteration     7 | Cost: 5.239113e+04
Iteration     8 | Cost: 4.771435e+04
Iteration     9 | Cost: 4.559863e+04
Iteration    10 | Cost: 4.385394e+04
Iteration    11 | Cost: 4.263562e+04
Iteration    12 | Cost: 4.184598e+04
Iteration    13 | Cost: 4.116751e+04
Iteration    14 | Cost: 4.073297e+04
Iteration    15 | Cost: 4.032577e+04
Iteration    16 | Cost: 4.009203e+04
Iteration    17 | Cost: 3.986428e+04
Iteration    18 | Cost: 3.971337e+04
Iteration    19 | Cost: 3.958890e+04
Iteration    20 | Cost: 3.949630e+04
Iteration    21 | Cost: 3.940187e+04
Iteration    22 | Cost: 3.934142e+04
Iteration    23 | Cost: 3.930822e+04
Iteration    24 | Cost: 3.926063e+04
Iteration    25 | Cost: 3.922334e+04
Iteration    26 | Cost: 3.920956e+04
Iteration    27 | Cost: 3.917145e+04
Iteration    28 | Cost: 3.914804e+04
Iteration    29 | Cost: 3.913479e+04
Iteration    30 | Cost: 3.910882e+04
Iteration    31 | Cost: 3.908992e+04
Iteration    32 | Cost: 3.908209e+04
Iteration    33 | Cost: 3.907380e+04
Iteration    34 | Cost: 3.906903e+04
Iteration    35 | Cost: 3.906437e+04
Iteration    36 | Cost: 3.905754e+04
Iteration    37 | Cost: 3.905112e+04
Iteration    38 | Cost: 3.904531e+04
Iteration    39 | Cost: 3.904023e+04
Iteration    40 | Cost: 3.903390e+04
Iteration    41 | Cost: 3.902800e+04
Iteration    42 | Cost: 3.902367e+04
Iteration    43 | Cost: 3.902195e+04
Iteration    44 | Cost: 3.902007e+04
Iteration    45 | Cost: 3.901780e+04
Iteration    46 | Cost: 3.901699e+04
Iteration    47 | Cost: 3.901489e+04
Iteration    48 | Cost: 3.901190e+04
Iteration    49 | Cost: 3.900929e+04
Iteration    50 | Cost: 3.900742e+04
Iteration    51 | Cost: 3.900630e+04
Iteration    52 | Cost: 3.900485e+04
Iteration    53 | Cost: 3.900348e+04
Iteration    54 | Cost: 3.900283e+04
Iteration    55 | Cost: 3.900208e+04
Iteration    56 | Cost: 3.900118e+04
Iteration    57 | Cost: 3.899982e+04
Iteration    58 | Cost: 3.899860e+04
Iteration    59 | Cost: 3.899710e+04
Iteration    60 | Cost: 3.899381e+04
Iteration    61 | Cost: 3.899242e+04
Iteration    62 | Cost: 3.899094e+04
Iteration    63 | Cost: 3.898986e+04
Iteration    64 | Cost: 3.898908e+04
Iteration    65 | Cost: 3.898811e+04
Iteration    66 | Cost: 3.898754e+04
Iteration    67 | Cost: 3.898736e+04
Iteration    68 | Cost: 3.898712e+04
Iteration    69 | Cost: 3.898687e+04
Iteration    70 | Cost: 3.898673e+04
Iteration    71 | Cost: 3.898634e+04
Iteration    72 | Cost: 3.898524e+04
Iteration    73 | Cost: 3.898369e+04
Iteration    74 | Cost: 3.898322e+04
Iteration    75 | Cost: 3.898257e+04
Iteration    76 | Cost: 3.898194e+04
Iteration    77 | Cost: 3.898141e+04
Iteration    78 | Cost: 3.898077e+04
Iteration    79 | Cost: 3.898025e+04
Iteration    80 | Cost: 3.897962e+04
Iteration    81 | Cost: 3.897909e+04
Iteration    82 | Cost: 3.897861e+04
Iteration    83 | Cost: 3.897734e+04
Iteration    84 | Cost: 3.897609e+04
Iteration    85 | Cost: 3.897533e+04
Iteration    86 | Cost: 3.897487e+04
Iteration    87 | Cost: 3.897466e+04
Iteration    88 | Cost: 3.897412e+04
Iteration    89 | Cost: 3.897387e+04
Iteration    90 | Cost: 3.897368e+04
Iteration    91 | Cost: 3.897350e+04
Iteration    92 | Cost: 3.897313e+04
Iteration    93 | Cost: 3.897298e+04
Iteration    94 | Cost: 3.897289e+04
Iteration    95 | Cost: 3.897284e+04
Iteration    96 | Cost: 3.897281e+04
Iteration    97 | Cost: 3.897270e+04
Iteration    98 | Cost: 3.897260e+04
Iteration    99 | Cost: 3.897253e+04
Iteration   100 | Cost: 3.897250e+04
```

```matlab:Code

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), num_users, num_features);

```

To predict the rating of movie <img src="https://latex.codecogs.com/gif.latex?\inline&space;i"/> for user<img src="https://latex.codecogs.com/gif.latex?\inline&space;j"/>, you need to compute <img src="https://latex.codecogs.com/gif.latex?\inline&space;(\theta^{(j)}&space;)^T&space;x^{(i)}"/> The code below computes the ratings for all the movies and users and displays the movies that it recommends (Figure 4), according to ratings that were entered earlier in the script. Note that you might obtain a different set of the predictions due to different random initializations.

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_3.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_3.png'
)

```matlab:Code
p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions,'descend');
for i=1:10
    j = ix(i);
    if i == 1
        fprintf('\nTop recommendations for you:\n');
    end
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), movieList{j});
end
```

```text:Output
Top recommendations for you:
Predicting rating 5.0 for movie Great Day in Harlem, A (1994)
Predicting rating 5.0 for movie Saint of Fort Washington, The (1993)
Predicting rating 5.0 for movie Someone Else's America (1995)
Predicting rating 5.0 for movie Santa with Muscles (1996)
Predicting rating 5.0 for movie Entertaining Angels: The Dorothy Day Story (1996)
Predicting rating 5.0 for movie Aiqing wansui (1994)
Predicting rating 5.0 for movie Prefontaine (1997)
Predicting rating 5.0 for movie They Made Me a Criminal (1939)
Predicting rating 5.0 for movie Marlene Dietrich: Shadow and Light (1996)
Predicting rating 5.0 for movie Star Kid (1997)
```

```matlab:Code
for i = 1:length(my_ratings)
    if i == 1
        fprintf('\n\nOriginal ratings provided:\n');
    end
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), movieList{i});
    end
end
```

```text:Output
Original ratings provided:
Rated 4 for Toy Story (1995)
Rated 3 for Twelve Monkeys (1995)
Rated 5 for Usual Suspects, The (1995)
Rated 4 for Outbreak (1995)
Rated 5 for Shawshank Redemption, The (1994)
Rated 3 for While You Were Sleeping (1995)
Rated 5 for Forrest Gump (1994)
Rated 2 for Silence of the Lambs, The (1991)
Rated 4 for Alien (1979)
Rated 5 for Die Hard 2 (1990)
Rated 5 for Sphere (1998)
```

# Submission and Grading

After completing various parts of the assignment, be sure to use the submit function system to submit your solutions to our servers. The following is a breakdown of how each part of this exercise is scored.

!['/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_4.png'
](README_images/'/MATLAB Drive/ex1-ex8-matlab/ex8/README_images/image_4.png'
)

You are allowed to submit your solutions multiple times, and we will take only the highest score into consideration.


