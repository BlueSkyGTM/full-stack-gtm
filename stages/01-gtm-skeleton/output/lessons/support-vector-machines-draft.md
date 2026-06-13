# Support Vector Machines

## Hook

You have a binary classification problem—leads that convert vs. leads that don't, accounts that churn vs. accounts that stay. Logistic regression draws a line and hopes for the best. SVM finds the line that maximizes the gap between your two classes, then guards that gap. That gap is your margin of safety.

## Concept

Explain the geometric mechanism: SVM locates the hyperplane that maximizes the minimum distance (margin) to the nearest points from either class. Those nearest points are the support vectors—they alone define the boundary. Cover the kernel trick as a dimensionality lift: when your data isn't linearly separable in its current space, a kernel implicitly maps it to a higher-dimensional space where a hyperplane works, without ever computing that mapping explicitly. Cover soft-margin parameter C as the tradeoff between margin width and misclassification tolerance.

## Build

Implement a binary SVM classifier using scikit-learn's `SVC`. Train on a small synthetic dataset with a nonlinear boundary using an RBF kernel. Print the support vector indices, the decision boundary coordinates, and classification accuracy. Show how changing C and gamma alters the boundary by printing results for two parameter sets side by side.

Exercise hooks:
- **Easy:** Train a linear SVM on a 2D dataset with two clearly separable clusters. Print the support vectors and accuracy.
- **Medium:** Switch from linear to RBF kernel on the same dataset after adding overlap. Print accuracy for both and compare.
- **Hard:** Write a loop over a grid of C and gamma values. Print the combination that yields the highest cross-validated accuracy.

## Use It

SVMs are binary classifiers at their core. In GTM, the canonical application is lead scoring as binary classification: convert / won't convert. This maps to the **Scoring & Prioritization** cluster in Zone 2. Build a pipeline where firmographic and behavioral features (company size, page visits, email opens) feed into an SVM. The margin gives you a confidence measure—points far from the boundary are high-confidence scores; points near the boundary are your "maybe" bucket worth routing to human review. Compare SVM output to a logistic regression baseline on the same feature set.

Exercise hooks:
- **Easy:** Train an SVM on a small lead dataset with two features. Print which leads fall within the margin (the "uncertain" bucket).
- **Medium:** Add feature scaling (StandardScaler) before the SVM. Print accuracy before and after scaling to demonstrate why SVMs are sensitive to feature magnitude.
- **Hard:** Train SVM and logistic regression on the same data. Print a side-by-side comparison of predictions on a held-out test set. Count where they disagree.

## Ship It

Production considerations: SVM prediction time scales with the number of support vectors, not the training set size—print `len(model.support_vectors_)` to see your inference budget. Kernel SVMs on large datasets (>50K rows) become expensive; print training times to establish your ceiling. Cover serialization with joblib, versioning your scaler and model together. Discuss what happens when the distribution shifts: SVM boundaries are fixed at train time, so new patterns require retraining.

Exercise hooks:
- **Easy:** Serialize a trained SVM pipeline (scaler + model) with joblib. Load it back and print predictions on a new sample to confirm round-trip integrity.
- **Medium:** Train an SVM on progressively larger subsets of a dataset (100, 1000, 10000 rows). Print training time and number of support vectors for each.
- **Hard:** Simulate distribution shift by training on one class balance (70/30) and testing on another (50/50). Print the accuracy drop and discuss whether SVM is the right tool for shifting distributions.

## Assess

Questions target mechanism, not trivia: Why do support vectors alone define the boundary? What does C actually control geometrically? Why does an RBF kernel separate nonlinear data without explicitly computing the higher-dimensional space? When would you choose SVM over logistic regression or a tree-based method—and when would you not?

Exercise hooks:
- **Easy:** Given a printed set of support vector indices, remove one and retrain. Print the new boundary and accuracy to observe the change.
- **Medium:** Explain in one paragraph why SVM performance degrades on imbalanced datasets, then demonstrate it by training on a 95/5 split and printing the confusion matrix.
- **Hard:** Implement a linear SVM from scratch using only numpy (gradient descent on the hinge loss). Print weights and bias, compare to sklearn's output on the same data.

---

**GTM Redirect:** Scoring & Prioritization (Zone 2). SVM is a binary classifier. Lead scoring is binary classification. The margin is your confidence signal.