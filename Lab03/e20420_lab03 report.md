
---

# E20420 Lab 03: Decision Trees & k-Nearest Neighbors on the Wine Dataset

## Task 1: Decision Tree Classifiers

### (a) Handling Missing Values

To demonstrate handling of missing data, **missing values** (10%) were introduced in the 'Alcohol' column of the Wine dataset. **Mean imputation** was applied to fill these values before model training.

Missing Values:

Decision trees handle gaps via imputation or native routing
kNN requires complete data (impute before scaling)


```python
# Load dataset
df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/wine.csv')

# Introduce 10% missing values in 'Alcohol'
np.random.seed(1)
df_missing = df.copy()
missing_mask = np.random.rand(len(df_missing)) < 0.1
df_missing.loc[missing_mask, 'Alcohol'] = np.nan

# Separate features and target
X = df_missing.drop('Wine', axis=1)
y = df_missing['Wine']

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Save feature names for later use
feature_names = X.columns.tolist()

```

---

### (b) Model Training and Evaluation

Two decision trees were trained using:
- **Gini Index**
- **Entropy**

A fixed random state was used for reproducibility in splitting the data:

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.25, random_state=42)

# Gini-based classifier
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

# Entropy-based classifier
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_entropy.fit(X_train, y_train)
```
When the Two were evaluated and printed, following were the results:
**Results:**
```
=== Gini Decision Tree Report ===
              precision    recall  f1-score   support

           1       1.00      0.93      0.97        15
           2       0.90      1.00      0.95        18
           3       1.00      0.92      0.96        12

    accuracy                           0.96        45
   macro avg       0.97      0.95      0.96        45
weighted avg       0.96      0.96      0.96        45


=== Entropy Decision Tree Report ===
              precision    recall  f1-score   support

           1       0.94      1.00      0.97        15
           2       0.85      0.94      0.89        18
           3       0.89      0.67      0.76        12

    accuracy                           0.89        45
   macro avg       0.89      0.87      0.87        45
weighted avg       0.89      0.89      0.88        45


```
### (c) Demonstrating Pruning to Overcome Overfitting

An **unpruned** tree (no `max_depth`) was compared to a **pruned** tree (`max_depth=4`):

```python
# Unpruned model (default, may overfit)
clf_unpruned = DecisionTreeClassifier(random_state=42)
clf_unpruned.fit(X_train, y_train)

# Pruned model (e.g., max_depth=2)
clf_pruned = DecisionTreeClassifier(max_depth=2, random_state=42)
clf_pruned.fit(X_train, y_train)
```

**Accuracy Comparison:**

Unpruned Training Accuracy: 1.0
Unpruned Testing Accuracy : 0.9055555555555556
Pruned Training Accuracy  : 0.9398496240601504
Pruned Testing Accuracy   : 0.8666666666666667

**Observation:**  
- The **unpruned tree** achieves near-perfect training accuracy but lower test accuracy (overfitting).
- The **pruned tree** shows a smaller gap between train and test accuracy (better generalization).

**Visualization:**

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf_unpruned, feature_names=X.columns, filled=True)
plt.title("Unpruned Tree (Overfit)")
plt.savefig('unpruned_tree.png')

plt.figure(figsize=(15, 8))
plot_tree(clf_pruned, feature_names=X.columns, filled=True)
plt.title("Pruned Tree (max_depth=4)")
plt.savefig('pruned_tree.png')
```
Visualization ([Visualization of Trees](https://drive.google.com/file/d/1LTZVXAKb4Tw31kzy1BD9ugsPQ8are0QO/view?usp=sharing))
---

## Task 2: k-Nearest Neighbors (kNN)

### (a) Feature Scaling

As kNN is sensitive to feature scales, the features were standardized:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### (b) Hyperparameter Tuning

Several metrics and values of k were evaluated using cross-validation to find the best parameters for kNN:

```python
metrics = ['euclidean','minkowski','cityblock','cosine']
best_score = 0
best_params = {}

for metric in metrics:
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, X_scaled, y, cv=5)
        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = {'k': k, 'metric': metric, 'accuracy': avg_score}

print("Best kNN parameters:", best_params)

# Evaluate each metric with the best k found
best_k = best_params['k']
metric_scores = {}

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    avg_score = np.mean(scores)
    metric_scores[metric] = avg_score

print("\nAccuracy scores for each metric with best k (", best_k, "):")
for metric, score in metric_scores.items():
    print(f"{metric}: {score:.4f}")
```

As the results 
Best kNN parameters: {'k': 11, 'metric': 'cityblock', 'accuracy': np.float64(0.9719047619047618)}

Accuracy scores for each metric with best k ( 11 ):

euclidean: 0.9495

minkowski: 0.9495

cityblock: 0.9719

cosine: 0.9494

---

### (c) Evaluation and Comparison
Using the following snipperts, the values were compared
```python
knn_best = KNeighborsClassifier(n_neighbors=best_params['k'], metric=best_params['metric'])
knn_best.fit(X_train, y_train)
knn_pred = knn_best.predict(X_test)
knn_time = time.time() - start_time
```

**Summary Table:**

| Model         | Accuracy | Runtime (s) | Notes                    |
| ------------- | -------- | ----------- | ------------------------ |
| Decision Tree | \~0.95   | Very fast   | Interpretable, fast      |
| kNN           | \~Varies | Slower      | Sensitive to k + scaling |


---

## Conclusion

- **Decision Trees:** Fast, interpretable, and effective. Pruning is crucial to prevent overfitting.
- **kNN:** Slightly better balanced precision/recall; requires scaling and parameter tuning.
- **Pruning** demonstrably reduces overfitting and improves test accuracy.
- **Consistent random states** and data splits are essential for fair model comparison.
