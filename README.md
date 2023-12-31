# My-Model-Techniques-
### Pipelining examples 

https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/

https://www.youtube.com/watch?v=HZ9MUzCRlzI  -  watch multiple exmaples here ..basic code available to brush up the topics 

https://scikit-learn.org/stable/auto_examples/ensemble/plot_feature_transformation.html#sphx-glr-auto-examples-ensemble-plot-feature-transformation-py

### Feature Importance
Feature Importance
Feature Importance that i have  and can be used post transformation steps (scalar, One hot, Normalization, resampling,Missing values...etc) and before training model ...

1. sklearns SelectKbest(chi2 ,mutual info ..etc ) c2 for linear correlation var vs target (pvalue, 2 variables are indepen)
  mutual info - information theory scoring - Non Linear correl )
Chi-Square or ANOVA Test: If your features are categorical, you can use statistical tests like the Chi-Square test or Analysis of Variance (ANOVA) to evaluate the relationship between each feature and the target variable. Features with significant p-values indicate higher importance.
  
3. sklearns permutation_importance

      "we need to be careful while selexting 1, 2 here because this just gives us top features but we might not need features with same score instead use non redundant ones, then we go for 5th option \n", 

4. sklearns feature_importances_ with diff models  randamForest - impurities, LogReg, DT...etc
Tree-based Feature Importance: Decision tree-based classifiers like Random Forests or Gradient Boosting models provide feature importance scores based on how much a feature reduces impurity in the trees. gain and Gini index are used to measure the usefulness of a feature for splitting the data. Higher values of information gain or lower values of the Gini index indicate more important features.
Coefficient Magnitude: For linear models such as Logistic Regression or Support Vector Machines (SVM), you can examine the magnitude of the coefficients assigned to each feature. Larger absolute values of coefficients indicate more important features.

6. Recursive Feature Elimination  - decription is written above 

7. remove redundant features (same importance)

5. Dimentional Reduction (  PCA )- Unsupervised, new features created - Not recommended if interested in same features- More columns
   also, L1 and L2 regularization can be applied to both classification and regression models. They are regularization techniques used to prevent overfitting and improve the generalization ability of machine learning models. this can be used futher for better model 



There are different ways to get feature importance for a model with too many columns. Here are a few common approaches:

1. Random Forest: One of the advantages of using Random Forest algorithm is that it provides a built-in feature importance metric based on the reduction in impurity achieved by each feature. You can access the feature importance scores using the `feature_importances_` attribute of the Random Forest model. Here's an example:


from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)

importances = rf.feature_importances_

The `importances` variable will contain an array of feature importances for each column in `X_train`.

2. Permutation Importance: Another approach is to use Permutation Importance to measure the importance of features. Permutation Importance works by permuting the values of each feature column in the dataset, and measuring the impact on the model's performance. Scikit-learn provides an implementation of this approach through the `PermutationImportance` class. Here's an example:


from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_train, y_train, n_repeats=10, random_state=0)

importances = result.importances_mean

The `importances` variable will contain an array of feature importances for each column in `X_train`.

3. Recursive Feature Elimination (RFE): RFE is an iterative approach that works by removing the least important features recursively until a desired number of features is reached. Scikit-learn provides a class called `RFECV` (Recursive Feature Elimination with Cross Validation) that performs RFE while using cross-validation to evaluate the model. Here's an example:


from sklearn.feature_selection import RFECV

rf = RandomForestRegressor(n_estimators=100)
rfecv = RFECV(estimator=rf, step=1, cv=10, scoring='r2')
rfecv.fit(X_train, y_train)

importances = rfecv.support_

The `importances` variable will contain a boolean mask indicating which columns are considered important by the `RFECV` algorithm.

These are just a few examples of how you can obtain feature importance for a high-dimensional dataset. Depending on your specific use case, you may need to experiment with different algorithms and approaches to understand the underlying structure of your data and identify the most important features.
### Feature scaling 
StandardScaler:

StandardScaler follows a standardization approach, also known as Z-score normalization.
It scales the features by subtracting the mean and dividing by the standard deviation of the feature values.
The resulting transformed values have a mean of 0 and a standard deviation of 1.
StandardScaler preserves the shape of the distribution but centers it around 0 and adjusts the spread.
It is suitable when the data does not have a specific range or when the distribution of the data is approximately Gaussian.
MinMaxScaler:

MinMaxScaler follows a normalization approach, also known as min-max scaling.
It scales the features by subtracting the minimum value and dividing by the range (maximum value - minimum value) of the feature values.
The resulting transformed values are in the range [0, 1].
MinMaxScaler compresses the data into a specific range, preserving the relative relationships between the data points.
It is suitable when the distribution of the data is not necessarily Gaussian and when the specific range of the data is important.
In summary, StandardScaler standardizes the data by centering it around 0 and adjusting the spread, while MinMaxScaler normalizes the data to a specific range, usually [0, 1]. The choice between the two depends on the nature of the data and the requirements of the specific machine learning algorithm /task
