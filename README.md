## Project Description
The Home Credit Group (HCG) is looking to develop a model with which they can determine the amount of risk associated with a particular loan. Such models are referred to as 'scorecards'. The main concern is that scorecards may need to be updated constantly based on customer behavior over time, which means that they may lack any kind of predictive ability. The goal is to develop a scorecard that can predict credit defaults prior to their occurence and be 'stable' over time. The data provided encompass a wide variety of categories, but no credit scores.

## Data Description

The data we are going to use (train/test) is provided by Home Credit Group and is available on Kaggle. The data is from a wide range of sources, including application forms, social-demographic data, previous credit behavior data, etc. Each data has a unique `case_id` which corresponds to one applicant. Also, the data with personal information and business sensitive information have been applied data masking but they still retain as much information value as possible.

In total, the data is around 26.77 GB and contains 32 separate files, including 465 features. For more information, please refer to the [data dictionary](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).

## Stakeholders
The HCG is the primary stakeholder in this project. They, along with other lending institutions, are clearly interested in making loans for which there is a low chance of default. Burrowers can also benefit from this analysis as it may allow a lender to offer loans or loans with better terms to individuals lacking an extensive credit history. 


## Key Performance Indicators (KPIs)
As mentioned in the project description, the main KPI is the ability to predict the likelihood of default, as well as the stability of the model over time. For accuracy part, the metric is AUC since this is a binary classification problem. For stability part, the metric is to evaluate the scorecard's performance over time. There are `WEEK_NUM` group and we calculate the AUC score for each group. Then a linear regression model $y = ax + b$ ($x$ is the `WEEK_NUM` and $y$ is the score) is used to evaluate the trend of AUC score over time. If the coefficient $a$ is negative, that is, the performance is dropping over time, then the final score of the model will be penalized.

Meanwhile, we calculate the standard deviation of the residual from the above linear regression model. If the standard deviation is too large, that is, the performance is not stable over time, then the final score of the model will be penalized.

The final score is a combination of the AUC score and the stability score:
$$
\text{Final Score} = \text{AUC Score} + 88 \cdot \min (0,a) - 0.5 \cdot \text{std(residual)}
$$