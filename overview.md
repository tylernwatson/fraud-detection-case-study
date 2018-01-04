# Overview

## Process Flow

1. Preprocess data
2. Fit Multinomial Naive Bayes model to Event Description text
3. Add probabilities of fraud from Naive Bayes model as a feature to original feature set
4. Fit all numeric features (including Naive Bayes probabilities) to a Random Forest Classifier
5. Predict!

## Preprocessing

- Add Fraud column -- we used all acct_type labels that included the word "Fraud" as fraudulent
- Strip HTML tags from Event Description
- Vectorize Event Descriptions usng TF-IDF
    - Removed stop words, max_df = .5
   

## Assessment Metrics Selected

Since this is an unbalanced dataset, we are using Precision and Recall as our main assessment metrics.

Multinomial Naive Bayes Metrics:

- MN Accuracy: 0.915760111576
- MN naive recall: 0.228
- MN naive precision: 0.61

Random Forest Metrics:

- RF Accuracy 0.988
- RF Recall: 0.914
- RF Precision: 0.939

## Validation and Testing Methodology

- Used train_test_split and cross_val_scores

## Parameter Tuning Involved in Generating the Model

Multinomial Naive Bayes Tuning Parameter:
alpha (laplace smoothing) = .01

## further steps you might have taken if you were to continue the project

- Further tune Random Forest Classifier
- Calculate / incorporate event revenue based on ticket_types column
- Further investigate for signal in other non-numeric features

