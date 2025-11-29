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


---

## The Data Science Workflow

Understanding the complete data science process is crucial. Here's how all the pieces fit together:

### 1. Data Collection and Cleaning
- **Collection:** Gather data from various sources (databases, APIs, files, web scraping)
- **Cleaning:** Handle missing values, remove duplicates, fix data types
- **Exploration:** Use visualizations (histograms, scatter plots, box plots) to understand data distribution
- **Quality checks:** Identify outliers, check for data integrity issues

### 2. Feature Engineering vs. Dimensionality Reduction (PCA)

These are fundamentally different approaches used at different stages:

**Feature Engineering (Before Modeling):**
- **What it is:** Manually selecting and creating features based on domain knowledge
- **Purpose:** Improve model performance by creating more informative features
- **Examples:**
  - Creating new features: `total_spending = price × quantity`
  - Encoding categorical variables: Converting "color" into separate binary columns
  - Scaling features: Normalizing values to 0-1 range
- **When to use:** When you understand your data and can create meaningful combinations
- **Advantage:** You control what features are created and why

**Dimensionality Reduction/PCA (After Feature Engineering):**
- **What it is:** Automatically reducing the number of features while preserving information
- **Purpose:** Handle high-dimensional data, reduce noise, improve computational efficiency
- **How it works:** Creates new features that are combinations of original features
- **When to use:** 
  - When you have too many features (curse of dimensionality)
  - For visualization (reduce to 2-3 dimensions)
  - To reduce noise in data
  - Before clustering or classification on high-dimensional data
- **Advantage:** Automatically finds optimal combinations without manual work

**Key Difference:** Feature engineering is **manual and interpretable**, while PCA is **automatic but creates abstract features** that are harder to interpret.

### 3. Train-Test Split: When and Why

**Why Split Data?**
- **Honest evaluation:** Test on data the model has never seen
- **Prevent overfitting:** Ensures model generalizes to new data, not just memorizes training data
- **Simulate real-world:** Mimics how model will perform on future, unseen data

**When to Split?**
```
1. Data Collection & Cleaning
2. Feature Engineering (create new features)
3. → SPLIT DATA HERE ← (75/25 or 80/20)
4. Apply PCA (fit on training data only!)
5. Train Model (using training data)
6. Evaluate Model (using test data)
```

**Critical Rule:** 
- **Fit transformations (scaling, PCA) ONLY on training data**
- Then apply those same transformations to test data
- Never let your test data "leak" into training process

**The Data Split Explained:**
- **X_train (75%):** Features for training - model learns from these
- **y_train (75%):** Labels for training - correct answers model learns from
- **X_test (25%):** Features for testing - model makes predictions
- **y_test (25%):** Labels for testing - used only to evaluate predictions

**Important:** The model sees both X_train AND y_train during training, but only X_test during prediction. We compare its predictions to y_test to measure accuracy.

### 4. Supervised vs. Unsupervised Learning

**Supervised Learning:**
- **What it is:** Learning from labeled data (you know the answers)
- **Data structure:** You have both features (X) and target labels (y)
- **Goal:** Learn to predict y from X
- **Examples:**
  - Regression: Predicting house prices (continuous values)
  - Classification: Predicting if email is spam (categories)
- **Algorithms:** Linear Regression, Logistic Regression, Decision Trees, k-NN, Random Forest

**Unsupervised Learning:**
- **What it is:** Finding patterns in data without labels (no right answers)
- **Data structure:** You only have features (X), no target labels (y)
- **Goal:** Discover hidden structure or patterns
- **Examples:**
  - Clustering: Grouping similar customers
  - Dimensionality Reduction: Reducing features while preserving information
- **Algorithms:** k-Means, Hierarchical Clustering, PCA

**Key Difference:** Supervised learning has a **teacher** (labels), unsupervised learning explores data **independently**.

### 5. Dependent vs. Independent Variables

**Independent Variables (Features, Predictors, X):**
- **What they are:** The input data you use to make predictions
- **Symbol:** Usually denoted as X (can be X₁, X₂, ..., Xₙ for multiple features)
- **Examples:**
  - House features: size, bedrooms, location, age
  - Patient data: age, blood pressure, cholesterol
  - Email features: word count, sender, time sent
- **Control:** These are the variables you have or can measure

**Dependent Variable (Target, Label, y):**
- **What it is:** The output you're trying to predict or understand
- **Symbol:** Usually denoted as y
- **Examples:**
  - House price (what you're predicting based on house features)
  - Disease diagnosis (what you're predicting based on patient data)
  - Email classification (spam/not spam based on email features)
- **Depends on:** Its value depends on the independent variables

**Easy way to remember:** 
- Independent variables → **What you know**
- Dependent variable → **What you want to predict**

---

## Understanding Model Evaluation

### Regression Evaluation Metrics

When predicting continuous values (prices, temperatures, etc.):

**1. R² (R-squared) - "How well does my model fit?"**
- **Range:** 0% to 100%
- **Interpretation:**
  - 0% = Model is just guessing (useless)
  - 50% = Model explains half the variance
  - 90% = Model explains most variance (excellent!)
  - 100% = Perfect predictions (rare in real life)
- **Example:** R² = 75% means "75% of house price differences are explained by the features I used"

**2. MAE (Mean Absolute Error) - "How far off am I, on average?"**
- **Units:** Same as your target variable
- **Interpretation:** The average size of your prediction errors
- **Example:** MAE = €20,000 for house prices means "on average, predictions are off by €20,000"
- **Advantage:** Easy to explain to non-technical people

**3. MSE (Mean Squared Error) - "How much do I punish big mistakes?"**
- **Calculation:** Square each error before averaging
- **Why square?** Big mistakes get punished much more
- **Example:** 
  - Error of €1,000 → Contributes 1,000² = 1,000,000
  - Error of €10,000 → Contributes 10,000² = 100,000,000
- **Use when:** Big errors are much worse than small ones

### Classification Evaluation Metrics

When predicting categories (spam/not spam, digit recognition):

**1. Accuracy - "What percentage did I get right?"**
- **Formula:** Correct predictions / Total predictions
- **Example:** 95% accuracy = correctly classified 95 out of 100 images
- **Limitation:** Can be misleading with imbalanced data

**2. Precision - "Of my positive predictions, how many were actually correct?"**
- **Formula:** True Positives / (True Positives + False Positives)
- **Example:** Email spam filter with 90% precision = 90% of emails marked as spam actually are spam
- **Use when:** False positives are costly

**3. Recall - "Of all actual positives, how many did I find?"**
- **Formula:** True Positives / (True Positives + False Negatives)
- **Example:** Disease detection with 85% recall = detected 85% of sick patients
- **Use when:** Missing positives is dangerous (medical diagnosis)

**4. F1-Score - "Balance between precision and recall"**
- **Formula:** Harmonic mean of precision and recall
- **Use when:** You need balance between precision and recall

**5. Confusion Matrix - "Where exactly am I making mistakes?"**
```
                Predicted
              No      Yes
Actual  No    TN      FP
        Yes   FN      TP
```
- **True Positive (TP):** Correctly predicted positive
- **True Negative (TN):** Correctly predicted negative
- **False Positive (FP):** Incorrectly predicted positive (Type I error)
- **False Negative (FN):** Incorrectly predicted negative (Type II error)

### Clustering Evaluation Metrics

When grouping similar items without labels:

**1. Inertia (Within-Cluster Sum of Squares)**
- **What it measures:** How tightly grouped each cluster is
- **Interpretation:** Lower is better (points closer to their cluster centers)
- **Limitation:** Always decreases as k increases

**2. Elbow Method - "How many clusters should I use?"**
- **How it works:** Plot inertia vs. number of clusters (k)
- **Look for:** The "elbow" - where adding more clusters doesn't help much
- **Visual approach:** The point where the graph bends sharply

**3. Silhouette Score**
- **Range:** -1 to 1
- **Interpretation:**
  - Close to 1: Well-separated clusters
  - Close to 0: Overlapping clusters
  - Negative: Points might be in wrong cluster

---

## Understanding Classification Models

### How Classification Works

**Core Concept:** Classification models draw **decision boundaries** in the feature space to separate different classes.

**Simple Example - Email Spam Classification:**
Imagine you have two features:
- X₁ = Number of exclamation marks
- X₂ = Number of money-related words

The model learns to draw a line (or curve) that separates spam from not spam:
```
    Money words
        ↑
    10 |  SPAM  SPAM    |
       |  SPAM  SPAM    | ← Decision Boundary
     5 |               |
       | Ham  Ham      |
     0 |_______________|______→
       0    5    10      Exclamation marks
```

### How Different Models Make Decisions

**Logistic Regression:**
- **Boundary:** Straight line (or hyperplane in higher dimensions)
- **How it works:** Calculates probability that an example belongs to each class
- **Output:** Probabilities (0 to 1)
- **Example:** "This email has 0.85 (85%) probability of being spam"
- **Good for:** Linearly separable data, when you need probabilities

**Decision Trees:**
- **Boundary:** Rectangular regions (axis-aligned splits)
- **How it works:** Asks yes/no questions about features
- **Example Questions:**
  1. "Does it have more than 5 exclamation marks?" → Yes
  2. "Does it mention money more than 3 times?" → Yes
  3. "Prediction: SPAM"
- **Good for:** Non-linear patterns, interpretable rules
- **Visualize:** Like a flowchart of decisions

**k-Nearest Neighbors (k-NN):**
- **Boundary:** Flexible, follows the natural grouping of data
- **How it works:** "You are like your neighbors"
  1. Find the k closest examples to the new point
  2. Let them "vote" on the class
  3. Majority wins
- **Example:** k=5, find 5 nearest emails → 4 are spam, 1 is ham → Predict: SPAM
- **Good for:** Complex, non-linear boundaries; when similar items truly belong to same class
- **Distance matters:** Uses Euclidean distance by default

### Classification Model Comparison

| Model | Decision Boundary | Interpretability | Speed | Best For |
|-------|------------------|------------------|-------|----------|
| Logistic Regression | Linear | High | Fast | Linear patterns, probabilities |
| Decision Trees | Rectangular | Very High | Fast | Non-linear, readable rules |
| k-NN | Flexible | Medium | Slow | Complex boundaries, similar=same |

---

## Deep Dive: Decision Trees and k-NN

### Decision Trees Explained

**How They're Built:**
1. **Start:** All data at root
2. **Find best split:** Which feature/threshold best separates classes?
3. **Split data:** Create two branches
4. **Repeat:** For each branch, find best split
5. **Stop when:** 
   - Node is pure (all same class)
   - Reached maximum depth
   - Too few samples to split

**Example - Digit Recognition:**
```
Root: All digits
├─ Pixel[32] < 8?
│  ├─ Yes → Likely 0 or 7
│  │  └─ Pixel[16] < 4?
│  │     ├─ Yes → Predict: 7
│  │     └─ No → Predict: 0
│  └─ No → Likely 1 or 8
│     └─ Pixel[48] < 12?
│        ├─ Yes → Predict: 1
│        └─ No → Predict: 8
```

**How They're Evaluated:**
- **Accuracy:** Percentage of correct predictions
- **Feature Importance:** Which features are used most in splits
- **Tree Depth:** Deeper = more complex = risk of overfitting

**How to Make Them Better:**
1. **Pruning:** Remove branches that don't improve performance
2. **Max Depth:** Limit how deep tree can grow
3. **Min Samples Split:** Require minimum samples before splitting
4. **Ensemble Methods:** Random Forest (multiple trees voting)

### k-Nearest Neighbors (k-NN) Explained

**How k-NN Works - Step by Step:**
1. **Store training data:** k-NN is "lazy" - it just remembers all training examples
2. **Get new point:** Receive a new example to classify
3. **Calculate distances:** Measure distance to all training points
4. **Find k nearest:** Select the k closest training points
5. **Vote:** These k neighbors vote on the class
6. **Predict:** Majority class wins

**Example - Digit Recognition with k=5:**
```
New image (unknown digit)
↓
Calculate distance to all 1,347 training images
↓
Find 5 closest images: [3, 3, 3, 8, 3]
↓
Vote: Four 3's, one 8
↓
Prediction: 3
```

**How Distance Affects k-NN:**
- **Distance metric:** Usually Euclidean: √[(x₁-x₂)² + (y₁-y₂)² + ...]
- **Small distances:** Points are very similar
- **Large distances:** Points are different
- **Key insight:** Closest neighbors have most influence

**Choosing k - The Critical Decision:**
- **k = 1:** Use only closest neighbor
  - **Pro:** Very flexible boundaries
  - **Con:** Sensitive to noise, overfitting
- **k = 5-10:** Common choice
  - **Pro:** Balanced
  - **Con:** Need to experiment
- **k = large:** Use many neighbors
  - **Pro:** Smooth boundaries, robust to noise
  - **Con:** May include irrelevant distant points, underfitting

**How k-NN is Evaluated:**
- **Accuracy:** Test on unseen data
- **Cross-validation:** Test multiple train-test splits to ensure robustness
- **Optimal k:** Try different k values, plot accuracy vs. k

**How to Make k-NN Better:**
1. **Feature scaling:** Normalize features so distances are meaningful
2. **Optimal k:** Use cross-validation to find best k
3. **Distance metric:** Try Manhattan or Minkowski distance
4. **Dimensionality reduction:** Use PCA first to reduce features
5. **Remove outliers:** They can distort neighborhoods

---

## Identifying and Preventing Overfitting

### What is Overfitting?

**Simple Definition:** Your model memorizes the training data instead of learning general patterns.

**Analogy:** Like a student who memorizes exam answers without understanding concepts - fails on new questions!

### How to Easily Identify Overfitting

**The Key Sign:**
```
Training accuracy: 99% 🎉
Testing accuracy:  65% 😱
→ OVERFITTING!
```

**Visual Signs:**
1. **Huge gap** between training and testing performance
2. **Training accuracy very high** (often >95%)
3. **Testing accuracy much lower**
4. **Model performs worse as training continues** (in iterative models)

**Example - Decision Tree Overfitting:**
```python
tree = DecisionTreeClassifier(max_depth=None)  # No limit!
tree.fit(X_train, y_train)

print(f"Training accuracy: {tree.score(X_train, y_train)}")  # 100%
print(f"Testing accuracy: {tree.score(X_test, y_test)}")     # 75%
# Gap of 25% = OVERFITTING
```

### Why Overfitting Happens

1. **Model too complex:** Too many parameters, too deep trees
2. **Too little data:** Not enough examples to learn from
3. **Training too long:** Model starts memorizing noise
4. **No regularization:** Nothing preventing over-complexity

### How to Prevent Overfitting

**1. More Training Data**
- Gives model more examples to learn from
- Harder to memorize when data is abundant

**2. Simpler Models**
- Decision Trees: Limit `max_depth`, set `min_samples_split`
- k-NN: Use larger k (more neighbors)
- Neural Networks: Fewer layers/neurons

**3. Regularization**
- Penalize model complexity
- Force model to focus on most important patterns

**4. Cross-Validation**
- Split data into multiple train-test combinations
- Ensures model performs well across different data splits
- More reliable than single train-test split

**5. Early Stopping**
- Stop training when validation performance stops improving
- Prevents over-learning on training data

**6. Ensemble Methods**
- Random Forest: Combines many trees, each slightly different
- Reduces overfitting of individual trees

### Underfitting vs. Overfitting vs. Just Right

```
UNDERFITTING          JUST RIGHT           OVERFITTING
(Too Simple)          (Goldilocks)         (Too Complex)

Training: 60%         Training: 85%        Training: 99%
Testing:  58%         Testing:  83%        Testing:  65%

Model is too basic    Perfect balance!     Model memorizes
Can't learn pattern   Generalizes well     Doesn't generalize
```

---

## Clustering vs. Classification: Key Differences

### The Fundamental Difference

**Classification (Supervised):**
- **Has labels:** You know the correct answers
- **Goal:** Learn to predict labels for new data
- **Example:** You have photos labeled "cat" or "dog" → Teach model to label new photos
- **Historical data:** Needs training data with known outcomes

**Clustering (Unsupervised):**
- **No labels:** You don't know the groupings beforehand
- **Goal:** Discover natural groups in data
- **Example:** You have customer data → Find groups of similar customers
- **Exploratory:** Discovering structure you didn't know existed

### Side-by-Side Comparison

| Aspect | Classification | Clustering |
|--------|---------------|------------|
| **Data** | Labeled (knows categories) | Unlabeled (no categories) |
| **Learning** | Supervised | Unsupervised |
| **Output** | Predicted class label | Cluster assignment |
| **Evaluation** | Compare to true labels | Measure cluster quality |
| **Use Case** | "Which class is this?" | "What groups exist?" |
| **Example** | Spam detection | Customer segmentation |

### When to Use Each

**Use Classification when:**
- You have labeled historical data
- You know what categories exist
- You want to predict categories for new data
- Examples: Medical diagnosis, fraud detection, image recognition

**Use Clustering when:**
- You have no labels
- You want to explore data structure
- You want to discover natural groupings
- Examples: Market segmentation, organizing documents, finding anomalies

### Can You Use Both Together?

**Yes!** Common workflow:
1. **Clustering first:** Discover groups in unlabeled data
2. **Label clusters:** Manually examine and label the discovered clusters
3. **Classification next:** Build classifier using the labeled clusters
4. **Predict:** Classify new data into these discovered categories

---

## How Dendrograms Work in Hierarchical Clustering

### What is a Dendrogram?

**Simple Definition:** A tree diagram showing how clusters are merged from bottom to top.

**Not a heatmap:** While both use colors and are visual, they show different things:
- **Heatmap:** Shows relationships between all pairs (correlation matrix)
- **Dendrogram:** Shows the order and distance of merges

### Reading a Dendrogram

```
Height (Distance)
    ↑
 10 |         ____________
    |        |            |
  8 |    ____|____    ____|____
    |   |         |  |         |
  5 |  _|_       _|_ |        _|_
    | |   |     |   ||       |   |
  0 | A   B     C   D E       F   G
    |_________________________→
              Observations
```

**Key Elements:**
1. **Bottom (leaves):** Individual data points
2. **Vertical lines:** Show distance/dissimilarity
3. **Horizontal lines:** Connect clusters being merged
4. **Height of merge:** How different the clusters are
5. **Top:** All points in one cluster

### How Clusters Are Made

**Step-by-Step Process:**

1. **Start:** Each point is its own cluster
   - 7 clusters: A, B, C, D, E, F, G

2. **Find closest pair:** Calculate distances between all clusters
   - A and B are closest (distance = 2)
   
3. **Merge:** Combine A and B
   - 6 clusters: {A,B}, C, D, E, F, G

4. **Repeat:** Find next closest pair
   - C and D are closest (distance = 3)
   
5. **Continue merging** until one cluster remains

**Result:** Complete hierarchy showing all possible numbers of clusters!

### Choosing the Number of Clusters

**The Cutting Method:**

Draw a horizontal line across the dendrogram:
```
Cut here (height = 8) → 3 clusters
    |         ____________
    |--------|------------|------- (cut line)
    |    ____|____    ____|____
```

**Where to cut?**
- **High cut:** Fewer, larger clusters
- **Low cut:** Many, smaller clusters
- **Best cut:** Largest vertical distance without horizontal lines crossing

### Distance Calculation Methods (Linkage)

**How to measure distance between clusters?**

1. **Single Linkage:** Distance between closest points
   - Pro: Can find elongated clusters
   - Con: Sensitive to outliers

2. **Complete Linkage:** Distance between farthest points
   - Pro: Makes compact clusters
   - Con: Can break large clusters

3. **Average Linkage:** Average distance between all pairs
   - Pro: Balanced approach
   - Con: Medium on everything

4. **Ward's Method:** Minimizes variance when merging
   - Pro: Creates evenly-sized, compact clusters
   - Con: Assumes spherical clusters
   - **Most commonly used!**

### Dendrogram vs. Heatmap

**Dendrogram:**
- Shows **hierarchical relationships**
- Displays **order of merging**
- Used for **clustering**
- Tree structure

**Heatmap:**
- Shows **all pairwise relationships**
- Displays **correlation or distance matrix**
- Used for **visualizing relationships**
- Grid structure

**Often used together:**
```
     Dendrogram
        |
    [Heatmap with reordered rows/columns]
```
The dendrogram orders the heatmap to group similar items!

---

## How Distance Affects Clustering

### Why Distance Matters

**Core Principle:** Clustering algorithms group points that are "close" together. But what does "close" mean?

### Types of Distance Metrics

**1. Euclidean Distance (Most Common)**
- **Formula:** √[(x₁-x₂)² + (y₁-y₂)² + ...]
- **Intuition:** Straight-line distance "as the crow flies"
- **Best for:** Continuous variables, when all dimensions equally important
- **Example:** 
  ```
  Point A: [1, 2]
  Point B: [4, 6]
  Distance: √[(4-1)² + (6-2)²] = √[9 + 16] = 5
  ```

**2. Manhattan Distance (City Block)**
- **Formula:** |x₁-x₂| + |y₁-y₂| + ...
- **Intuition:** Distance walking along a grid (like city streets)
- **Best for:** When movement restricted to axes
- **Example:**
  ```
  Point A: [1, 2]
  Point B: [4, 6]
  Distance: |4-1| + |6-2| = 3 + 4 = 7
  ```

**3. Cosine Distance**
- **What it measures:** Angle between vectors, not magnitude
- **Best for:** Text analysis, when direction matters more than magnitude
- **Example:** Comparing documents based on word frequencies

### Impact on Clustering

**Different distances → Different clusters:**

```
Euclidean Distance:          Manhattan Distance:
    o  o                         o  o
  o      o                     o      o
    o  o                         o  o
(Circular clusters)          (Square clusters)
```

### Feature Scaling is Critical!

**Problem without scaling:**
```
Feature 1: Income (range: $20,000 - $200,000)
Feature 2: Age (range: 20 - 65)

Distance dominated by income!
→ Age has almost no influence
```

**Solution - Scale features:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**After scaling:**
- All features have similar ranges
- Each feature contributes fairly to distance
- Clustering results much better!

### Distance in Different Algorithms

**k-Means:**
- Uses Euclidean distance to cluster centers
- Assumes spherical clusters
- Sensitive to feature scaling

**Hierarchical Clustering:**
- Can use any distance metric
- Distance affects shape of dendrogram
- Ward's method minimizes within-cluster variance

**DBSCAN:**
- Uses epsilon distance threshold
- Finds clusters of arbitrary shape
- Points within epsilon distance are neighbors

---

## Understanding Cross-Validation

### What is Cross-Validation?

**Problem:** Single train-test split might be lucky or unlucky
- Maybe test data was easier than typical?
- Maybe we accidentally got the "best" split?

**Solution:** Test multiple times with different data splits!

### How Cross-Validation Works

**k-Fold Cross-Validation (Most Common):**

```
Original Data: [All your data]
                    ↓
Split into 5 folds: |1|2|3|4|5|

Round 1: |T|T|T|T|V|  → Train on 1-4, Validate on 5
Round 2: |T|T|T|V|T|  → Train on 1-3,5, Validate on 4
Round 3: |T|T|V|T|T|  → Train on 1-2,4-5, Validate on 3
Round 4: |T|V|T|T|T|  → Train on 1,3-5, Validate on 2
Round 5: |V|T|T|T|T|  → Train on 2-5, Validate on 1

T = Training data
V = Validation data

Average the 5 scores → Final performance estimate
```

### Why Cross-Validation is Better

**Single Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # One score: 85%
# Is this good luck or truly representative?
```

**Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
# Five scores: [83%, 87%, 84%, 86%, 85%]
mean_score = scores.mean()  # 85%
std_score = scores.std()    # ±1.5%
# Much more confident in this estimate!
```

### Types of Cross-Validation

**1. k-Fold CV (Standard)**
- Split data into k equal parts
- Common choices: k=5 or k=10
- Good balance between computation and reliability

**2. Stratified k-Fold**
- Maintains class distribution in each fold
- **Important for imbalanced data**
- Example: If 10% of data is positive class, each fold has 10% positive

**3. Leave-One-Out (LOO)**
- k = n (number of samples)
- Each sample used as test once
- Very thorough but very slow
- Only for small datasets

**4. Time Series CV**
- For time-dependent data
- Always train on past, test on future
- Never train on future data!

### When to Use Cross-Validation

**Use CV for:**
- **Model selection:** Choosing between different algorithms
- **Hyperparameter tuning:** Finding best parameters
- **Performance estimation:** Getting reliable accuracy estimate
- **Small datasets:** Need to use all data efficiently

**Don't need CV when:**
- Very large datasets (single split sufficient)
- When doing final model evaluation (use separate held-out test set)
- When computation is extremely expensive

### Best Practices

**The Full Workflow:**
```
1. Split data: Training Set (80%) + Final Test Set (20%)
2. Use Training Set for cross-validation:
   - Try different models
   - Tune hyperparameters
   - Select best model
3. Train final model on ALL training data
4. Evaluate once on Final Test Set
5. Report this score
```

**Important Rules:**
- Never use test data in cross-validation
- Don't tune based on test set performance
- Keep test set completely hidden until final evaluation
- Report both mean and standard deviation of CV scores

### Cross-Validation Example

```python
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Try different max_depth values
for depth in [3, 5, 10, 15, 20]:
    model = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Depth {depth}: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Output:
# Depth 3:  0.850 (+/- 0.023)  ← Underfitting
# Depth 5:  0.920 (+/- 0.015)  
# Depth 10: 0.950 (+/- 0.012)  ← Best!
# Depth 15: 0.935 (+/- 0.045)  ← Starting to overfit (high variance)
# Depth 20: 0.925 (+/- 0.078)  ← Overfitting (very high variance)
```

---

## Additional Important Concepts

### Comparing Heat Maps and Correlation

**Heat Map:**
- Visual representation of a matrix
- Colors represent values (light to dark, cold to hot)
- Can show: correlations, confusion matrices, distance matrices, cluster assignments
- **Purpose:** Make patterns in matrices easy to see

**Correlation Heat Map Specifically:**
- Shows how features relate to each other
- Values from -1 (negative correlation) to +1 (positive correlation)
- **Dark colors:** Strong correlation
- **Light colors:** Weak correlation
- **Use:** Feature selection - remove highly correlated features (redundant)

**Example Use in Assignment:**
```python
# Create correlation matrix
correlation_matrix = df.corr()

# Visualize with heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Find highly correlated features (bonus!)
high_corr = (correlation_matrix.abs() > 0.9) & (correlation_matrix != 1.0)
# Remove redundant features
```

### When to Apply PCA in Your Workflow

```
Data Science Pipeline with PCA:

1. Collect Data
2. Clean Data (handle missing values, outliers)
3. Explore Data (visualizations, statistics)
4. Feature Engineering (create new features)
5. Feature Selection (remove irrelevant features)
   ↓
6. → APPLY PCA HERE ← (if needed)
   ↓
   When to use PCA:
   - Many features (>50)
   - Features are correlated
   - Computational constraints
   - Visualization needed
   ↓
7. Split Data (train-test split)
   ⚠️ Important: Fit PCA on training data only!
   ↓
8. Scale Features (StandardScaler)
9. Train Model
10. Evaluate Model
```

**PCA Decision Tree:**
```
Do you have >50 features? 
├─ No → Skip PCA, use original features
└─ Yes → Are features highly correlated?
    ├─ No → Consider feature selection instead
    └─ Yes → Is interpretability critical?
        ├─ Yes → Use feature selection (keep original features)
        └─ No → Use PCA! (create new combined features)
```

### Take-Home Assignment Bonus Tips

**Heat Map Analysis Bonus Points:**
1. **Identify correlations:** Which features are highly correlated?
2. **Explain redundancy:** Why might correlated features be redundant?
3. **Feature selection:** Suggest which features to remove
4. **Multicollinearity:** Explain how this affects model performance
5. **Domain insight:** What do correlations tell you about the data?

**Example bonus answer:**
> "The heat map reveals that features X1 and X2 have 0.95 correlation, suggesting 
> they contain redundant information. Removing X2 would reduce dimensionality without 
> losing much information, potentially improving model performance and reducing 
> computation time. This multicollinearity could also cause instability in linear 
> regression coefficients."

---

## Quick Reference Guide

### Algorithm Selection Cheatsheet

**Need to predict a NUMBER?** → Regression
- Linear pattern? → Linear Regression
- Non-linear pattern? → Polynomial Regression

**Need to predict a CATEGORY?** → Classification
- Want probabilities? → Logistic Regression
- Want interpretable rules? → Decision Trees
- Have complex boundaries? → k-NN or Random Forest
- Small dataset? → k-NN
- Large dataset? → Logistic Regression or Random Forest

**Have NO LABELS?** → Unsupervised Learning
- Find groups? → Clustering (k-Means or Hierarchical)
- Reduce dimensions? → PCA
- Explore structure? → Start with PCA for visualization

**Too many features?** → Dimensionality Reduction
- Need interpretability? → Feature Selection
- Maximum performance? → PCA
- Then proceed with classification/regression

### Common Pitfalls to Avoid

1. **Don't test on training data** - Always use separate test set
2. **Don't forget to scale features** - Especially for k-NN and PCA
3. **Don't tune based on test set** - Use cross-validation on training data
4. **Don't ignore overfitting signs** - Monitor train vs. test performance
5. **Don't blindly trust high accuracy** - Check confusion matrix
6. **Don't apply transformations after split** - Fit on train, transform both
7. **Don't use PCA before understanding data** - Explore first!
8. **Don't forget domain knowledge** - Numbers aren't everything

---