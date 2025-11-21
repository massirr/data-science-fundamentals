# Data Science Fundamentals Course

This repository contains notebooks and exercises covering fundamental machine learning concepts and algorithms.

## Course Overview

The course is divided into two main sections: **Supervised Learning** (Chapters 1-6) and **Unsupervised Learning** (Chapters 7+).

---

## Part 1: Supervised Learning

In supervised learning, we have both input features (X) and known output labels (y). The goal is to learn a mapping from X to y.

### Regression (Predicting Continuous Values)

**Chapter 2: Linear Regression**
- Predict continuous numerical values
- Learn relationships between variables
- Example: Predicting house prices based on features
- Key concepts: Line of best fit, R², MSE

**Chapter 3: Polynomial Regression**
- Extend linear regression to capture non-linear relationships
- Add polynomial features (x², x³, etc.)
- Example: Predicting car performance based on specs
- Key concepts: Feature engineering, overfitting vs underfitting

### Classification (Predicting Categories/Labels)

**Chapter 4: Logistic Regression**
- Predict binary outcomes (Yes/No, True/False)
- Example: Predicting diabetes diagnosis
- Key concepts: Probability thresholds, accuracy, precision, recall, confusion matrix

**Chapter 5: Decision Trees**
- Create tree-like models for classification
- Easy to visualize and interpret
- Example: Wine quality classification
- Key concepts: Tree depth, splitting criteria, feature importance

**Chapter 6: k-Nearest Neighbors (k-NN)**
- Classify based on similarity to nearby points
- "Lazy learning" - no explicit model building
- Example: Mine vs Rock detection with sonar
- Key concepts: Distance metrics, choosing k, cross-validation

---

## Part 2: Unsupervised Learning

In unsupervised learning, we only have input features (X) without labels. The goal is to discover hidden patterns and structure in the data.

### Clustering (Finding Natural Groups)

**Chapter 7: k-Means Clustering**
- Partition data into k distinct clusters
- Group similar observations together
- Example: Handwritten digit recognition without labels
- Key concepts: Cluster centers, inertia, elbow method, within-cluster variance
- Real-world applications: Customer segmentation, image compression, document classification

**Chapter 8: Hierarchical Clustering**
- Build a hierarchy of clusters (tree-like structure)
- No need to specify k in advance
- Example: Grouping similar items or observations
- Key concepts: Dendrograms, linkage methods (single, complete, average), cutting the tree
- Real-world applications: Taxonomy creation, gene expression analysis, social network analysis

---

## Key Concepts

### Variable Types

**Categorical Variables:**
- Represent categories/labels
- Examples: Mine/Rock, Yes/No, Red/Blue/Green, digit labels (0-9)

**Continuous Variables:**
- Represent numerical measurements
- Examples: temperature, price, age, pixel intensities

### When to Use Each Algorithm

**Regression (Continuous Target):**
- Linear Regression: Simple linear relationships
- Polynomial Regression: Non-linear relationships

**Classification (Categorical Target):**
- Logistic Regression: Binary classification, probability estimates
- Decision Trees: Interpretable rules, handles non-linear patterns
- k-NN: Instance-based learning, good for complex boundaries
- Random Forest: Ensemble of trees, robust and accurate

**Clustering (No Target/Labels):**
- k-Means: Fast, scalable, works well with spherical clusters
- Hierarchical: Reveals nested structure, no need to specify k

### Model Evaluation

**For Regression:**
- R² (R-squared): How well the model fits the data
- MSE (Mean Squared Error): Average squared difference between predictions and actuals
- MAE (Mean Absolute Error): Average absolute difference

**For Classification:**
- Accuracy: Overall percentage of correct predictions
- Precision: Of positive predictions, how many were correct?
- Recall: Of actual positives, how many did we detect?
- F1-Score: Balance between precision and recall
- Confusion Matrix: Breakdown of all prediction types

**For Clustering:**
- Inertia: Within-cluster sum of squares (lower is better)
- Silhouette Score: How similar objects are to their own cluster vs other clusters
- Elbow Method: Visual approach to finding optimal k
- Dendrogram: Visual hierarchy of clusters (hierarchical clustering)
