# Module 12: Supervised Learning

## Overview of the Analysis

For this analysis we use supervised learning techniques to predict whether any particular loan is a `healthy` or `unhealthy` loan.

* For this analysis we use supervised learning techniques to predict whether any particular loan is a `healthy` or `unhealthy` loan.
* The data used for prediction includes information about the loan, the borrower, and the lending team's judgment of the loan overall.
* The data has an imbalance, with a higher number of healthy loans than unhealthy loans.
* We first use the data as-is and then resample using the oversampling approach in an attempt to address the imbalance mentioned above.

## Results

* Machine Learning Model 1 (Original Data):
```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384
```


* Machine Learning Model 2 (Resampled Data):
```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384
```

## Summary

Similar to the model with unsampled data it performs extremely well for both healthy and high-risk loans. Comparatively, the resampled model has gained performance both in recall at a slight loss in precision for the high-risk loans category. Overall, the resampled model is likely to perform better than the original model in production as it more accurately predicts unhealthy loans.

---

## Technologies

This application uses python 3, please install the necessary packages as described below to recreate the analysis.

---

## Installation Guide

```
pip install -r requirements.txt
```

---

## Usage

Open the file `credit_risk_resampling.ipynb` in a Jupyter notebook/lab environment to interact with the analysis.

---

## Contributors

[rowrowrowrow](https://github.com/rowrowrowrow)

---

## License

No license provided, you may not use the contents of this repo.
