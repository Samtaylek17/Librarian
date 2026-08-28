# scikit-learn__scikit-learn-14629  (27B / BOAD)

**Failure mode:** definition-anchoring  
**Note:** same, 27B

- base (no Librarian) `boad_seed0`: passed=True
- +Librarian `boad_librarian_seed0`: passed=False

**Gold/base edited files:** ['sklearn/model_selection/tests/test_validation.py', 'sklearn/multioutput.py', 'sklearn/tests/test_multioutput.py']

**+Librarian edited files:** ['sklearn/model_selection/_validation.py']

## Issue

```
AttributeError with cross_val_predict(method='predict_proba') when using MultiOuputClassifier
#### Description
I believe there is a bug when using `cross_val_predict(method='predict_proba')` with a `MultiOutputClassifer`. 

I think the problem is in the use of `estimator.classes_` here:
https://github.com/scikit-learn/scikit-learn/blob/3be7110d2650bbe78eda673001a7adeba62575b0/sklearn/model_selection/_validation.py#L857-L866

To obtain the `classes_` attribute of a `MultiOutputClassifier`, you need `mo_clf.estimators_[i].classes_` instead.

If core team members have any idea of how to address this, I am happy to submit a patch. 

#### Steps/Code to Reproduce

```python
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict

X, Y = make_multilabel_classification()

mo_lda = MultiOutputClassifier(LinearDiscriminantAnalysis())
pred = cross_val_predict(mo_lda, X, Y, cv=5) # Works fine
pred_proba =  cross_val_predict(mo_lda, X, Y, cv=5, method='predict_proba') # Returns error

```


#### Expected Results
Array with prediction probabilities.

#### Actual Results
```python
AttributeError: 'MultiOutputClassifier' object has no attribute 'classes_'
```

#### Versions
System:
    python: 3.6.8 |Anaconda, Inc.| (default, Feb 21 2019, 18:30:04) [MSC v.1916 64 bit (AMD64)]
executable: C:\Users\nak142\Miniconda3\envs\myo\python.exe
   machine: Windows-10-10.0.17134-SP0

```

## Librarian submit_result answers (the search guidance)

1. The `_enforce_prediction_order` function is defined in `/testbed/sklearn/model_selection/_validation.py` at lines 881–935.

## base (PASS) patch

```diff
diff --git a/sklearn/model_selection/tests/test_validation.py b/sklearn/model_selection/tests/test_validation.py
index 6fa2e4fee..7c4ea5dd4 100644
--- a/sklearn/model_selection/tests/test_validation.py
+++ b/sklearn/model_selection/tests/test_validation.py
@@ -1510,6 +1510,32 @@ def test_cross_val_predict_with_method_multilabel_rf_rare_class():
             check_cross_val_predict_multilabel(est, X, y, method=method)
 
 
+def test_cross_val_predict_with_method_multioutput_classifier():
+    # Test that cross_val_predict works with MultiOutputClassifier
+    # and method='predict_proba'. This tests the classes_ attribute
+    # of MultiOutputClassifier.
+    from sklearn.multioutput import MultiOutputClassifier
+    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
+
+    X, y = make_multilabel_classification(n_samples=100, n_labels=3,
+                                          n_classes=2, n_features=5,
+                                          random_state=42)
+    est = MultiOutputClassifier(LinearDiscriminantAnalysis())
+
+    # Test that classes_ attribute is accessible after fitting
+    est.fit(X, y)
+    assert hasattr(est, 'classes_')
+    assert isinstance(est.classes_, list)
+    assert len(est.classes_) == y.shape[1]
+
+    # Test cross_val_predict with predict_proba
+    pred_proba = cross_val_predict(est, X, y, cv=3, method='predict_proba')
+    assert isinstance(pred_proba, list)
+    assert len(pred_proba) == y.shape[1]
+    for p in pred_proba:
+        assert p.shape[0] == X.shape[0]
+
+
 def get_expected_predictions(X, y, cv, classes, est, method):
 
     expected_predictions = np.zeros([len(y), classes])
diff --git a/sklearn/multioutput.py b/sklearn/multioutput.py
index 463b72d40..5ff0e5de3 100644
--- a/sklearn/multioutput.py
+++ b/sklearn/multioutput.py
@@ -320,11 +320,19 @@ class MultiOutputClassifier(MultiOutputEstimator, ClassifierMixin):
     ----------
     estimators_ : list of ``n_output`` estimators
         Estimators used for predictions.
+
+    classes_ : array-like, shape (n_outputs, )
+        The class labels for each output.
     """
 
     def __init__(self, estimator, n_jobs=None):
         super().__init__(estimator, n_jobs)
 
+    @property
+    def classes_(self):
+        """Class labels for each output."""
+        return [estimator.classes_ for estimator in self.estimators_]
+
     def predict_proba(self, X):
         """Probability estimates.
         Returns prediction probabilities for each class of each output.
diff --git a/sklearn/tests/test_multioutput.py b/sklearn/tests/test_multioutput.py
index e70048a7f..e8b2bd998 100644
--- a/sklearn/tests/test_multioutput.py
+++ b/sklearn/tests/test_multioutput.py
@@ -276,6 +276,25 @@ def test_multi_output_classification():
                            list(predict_proba[i]))
 
 
+def test_multi_output_classification_classes():
+    # test that MultiOutputClassifier has a classes_ attribute
+    # which is a list of the classes_ of each estimator
+    forest = RandomForestClassifier(n_estimators=10, random_state=1)
+    multi_target_forest = MultiOutputClassifier(forest)
+
+    multi_target_forest.fit(X, y)
+
+    # Check that classes_ attribute exists and is a list
+    assert hasattr(multi_target_forest, 'classes_')
+    assert isinstance(multi_target_forest.classes_, list)
+    assert len(multi_target_forest.classes_) == n_outputs
+
+    # Check that classes_ matches the classes_ of each estimator
+    for i, estimator in enumer
```

## +Librarian (FAIL) patch

```diff
diff --git a/sklearn/model_selection/_validation.py b/sklearn/model_selection/_validation.py
index 95669a212..7da26b359 100644
--- a/sklearn/model_selection/_validation.py
+++ b/sklearn/model_selection/_validation.py
@@ -855,8 +855,19 @@ def _fit_and_predict(estimator, X, y, train, test, verbose, fit_params,
     predictions = func(X_test)
     if method in ['decision_function', 'predict_proba', 'predict_log_proba']:
         if isinstance(predictions, list):
+            # For multi-output estimators (e.g., MultiOutputClassifier),
+            # classes_ attribute is not directly available on the estimator
+            # but on each internal estimator
+            if hasattr(estimator, 'classes_'):
+                classes = estimator.classes_
+            elif hasattr(estimator, 'estimators_'):
+                classes = [est.classes_ for est in estimator.estimators_]
+            else:
+                raise AttributeError(
+                    "Estimator has no 'classes_' attribute and no "
+                    "'estimators_' attribute to access classes from.")
             predictions = [_enforce_prediction_order(
-                estimator.classes_[i_label], predictions[i_label],
+                classes[i_label], predictions[i_label],
                 n_classes=len(set(y[:, i_label])), method=method)
                 for i_label in range(len(predictions))]
         else:
```
