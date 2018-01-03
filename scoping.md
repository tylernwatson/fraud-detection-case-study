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

We will continue our preprocessing by 

* What models do you want to try?



* What metric will you use to determine success?


