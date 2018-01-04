#Fraud Detection

Before you get cranking on your model, take time to think about how to approach the problem.

* Think of what preprocessing you might want to do. How will you build your feature matrix? What different ideas do you have?

We began our EDA by creating a fraud column, using positive labels for the acct_types 'fraudster_event', 'fraudster',
and 'fraudster_att'. 

Further EDA showed that fraudulent events often have the following:
* Shorter event descriptions
* Smaller user_age

We also found that fraudulent events have a lower average payout (perhaps an attempt to avoid detection).

Furthermore, all accounts paying in Mexican pesos are fraudulent.

We will continue our preprocessing by lemmatizing, and tf-idf vectorizing the 'description' column.

* What models do you want to try?

We will do some feature regularization/reduction by using ridge and lasso regression. If these models perform well, we
will pass these regularized coefficients into the following models to see if they are more performant:

Logistic regression, support vector machines, random forest classifier, and gradient boosted trees.

We will also use multinomial Naive Bayes on the tfidf values to predict probabilities of each event being fraudulent.
If this is performing well, we are going to use these probabilities as a new feature and combine it with the most
predictive features we uncovered in our other models.

* What metric will you use to determine success?

Recall/sensitivity, which is our True Positive / Condition Positive. This will allow us to optimize for catching cases
of fraud.
