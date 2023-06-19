# My-Model-Techniques-
Feature Importance
Feature Importance that i have  and can be used post transformation steps (scalar, One hot, Normalization, resampling,Missing values...etc) and before training model ...

1. sklearns SelectKbest(chi2 ,mutual info ..etc ) c2 for linear correlation var vs target (pvalue, 2 variables are indepen)
  mutual info - information theory scoring - Non Linear correl )
  
2. sklearns permutation_importance

      "we need to be careful while selexting 1, 2 here because this just gives us top features but we might not need features with same score instead use non redundant ones, then we go for 5th option \n", 

3. Models sklearns feature_importances_ with diff models  randamForest - impurities, LogReg, DT...etc 

4. Recursive Feature Elimination  - decription is written above 

5. remove redundant features (same importance)

5. Dimentional Reduction (  PCA )- Unsupervised, new features created - Not recommended if interested in same features- More columns 
