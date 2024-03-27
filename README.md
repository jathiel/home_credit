## Project Description
The Home Credit Group (HCG) is looking to develop a model that accurately predicts the riskiness of a particular loan. Such models are referred to as 'scorecards'. Since scorecards need to be updated according to changes in consumer behavior over time, their predictive ability may change over time as well. The goal of this project is to develop a scorecard that can provide time-stable predictions of credit defaults. The data provided encompass a wide variety of categories, but no credit scores.

## Data Description

The data we are going to use (train/test) is provided by Home Credit Group and is available on Kaggle. The data were derived from various sources, including application forms, social-demographic data, previous credit behavior data, etc. Each data has a unique `case_id` which corresponds to one applicant. Moreover, personal information and business sensitive information are "masked" but still retain as much information as possible.

In total, the data is around 26.77 GB and contains 32 separate files, including 465 features. For more information, please refer to the [data dictionary](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).

## Stakeholders
The HCG is the primary stakeholder in this project. They, along with other lending institutions, have a strong financial incentive tp make loans that are unlikely to default. Borrowers may also benefit from improved predictions, as lenders may be more inclined to offer preferable loan terms to individuals without extensive credit history if other indicators (i.e., income and demographics) suggest a low chance of default. Improved loan predictions would also benefit organizations that supply consumer products that often require loans. Car dealerships, for instance, may increase their profits if previously ineligible lendees are given loans. 



## Key Performance Indicators (KPIs)
As mentioned in the project description, the main KPIs are the model's ability to accurately predict the likelihood of default and the stability of the model over time. AUC is the primary metric used to assess accuracy, meaning that the model(s) will aim to solve a binary classification problem. In terms of model stability over time, the key metric will be the scorecard's performance consistency over time. Using the `WEEK_NUM` group, we can calculate the AUC score for each group over time. A linear regression model $y = ax + b$ ($x$ is the `WEEK_NUM` and $y$ is the score) will be used to evaluate the trend of the AUC score over time. A negative coefficient $a$ will indicate that predictive performance is dropping over time and the final score of the model will be penalized.

We will also calculate the standard deviation of the residual from the above linear regression model. If the standard deviation is too large, the performance is not stable over time. If that is the case, then the final score of the model will be penalized.

The final score is a combination of the AUC score and the stability score:

$
\text{Final Score} = \text{AUC Score} + 88 \cdot \min (0,a) - 0.5 \cdot \text{std(residual)}
$

## References