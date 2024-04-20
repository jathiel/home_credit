# Notes on the data

After looking over the starter notebook and reading the discussion pages, we've learned the following:
  * This is a binary classification task where each case_id represents one observation. The target value of 0 or 1 in the main document states whether or not that loan experience a default. All of the cases are for individuals whose loans were approved.
  * The main metric here involves stability over the long-term. Being able to make accurate predictions in the short-run seems to be of no interest to the competition host. They want a model that will not experience a severe decline in performance over a long stretch of time.
  * The test data set is very small, but they have a hidden one that is much larger. Due to the number of cases and features available, memory issues will be something with which we must deal. Batches should be used.
  * Column name ends: 
## Relevant columns:
  * train_base: case_id, date_decision, WEEK_NUM, target
  * 
# Tasks

### Joining the Data

The home-credit-data-cleaning.ipynb document contains a lot of helpful information regarding how to put the data together. We should use this as a blueprint when cleaning the data ourselves. Be careful not to discard all NaN entries as they may indicate something of value depending on the feature.

## Model

This is a binary classification task where most of the features are categorical. An appropriate model needs to be selected.

### Memory Issues

The size of the data set means that we must use some kind of batch system when modeling. 

### Metric

A better understanding of the metric is needed. This may help steer feature and model selection.

### Vectorization
Columns need to be vectorized into numerical values for training models
ColumnTransformer