# Electricity Demand Prediction Model :-
This notebook loads the UCI Household Power Consumption dataset, performs exploratory data analysis, and trains machine learning models to predict power intensity based on time and sub-metering features.

"Regression analysis is a statistical method used to model the relationship between a dependent variable (e.g., energy consumption) and one or more independent variables (e.g., temperature, time of day, occupancy). It's a powerful tool for predicting future energy consumption based on historical data and understanding the factors that influence it."

PROBLEM STATEMENT :
Can we predict future household energy consumption from historic data?

# BACKGROUND :-

(1.) Core Concept Statistical method to model relationships between variables Predicts a dependent variable (target) based on independent variables (features) Shows both correlation and direction of relationships

(2.) Types of Regression

#Simple Linear: One feature → one target

y = mx + b

where y is target, x is feature, m is slope, b is intercept

#Multiple Linear: Multiple features → one target

y = b0 + b1x1 + b2x2 + ... + bnxn

#Polynomial: Nonlinear relationships

y = b0 + b1x + b2x² + b3x³

#Other Types: Logistic (for classification) Ridge (L2 regularization) Lasso (L1 regularization) Random Forest (ensemble method)

(3.)Key Components: Dependent Variable - What you're trying to predict Independent Variables - Features used for prediction Coefficients - Weights showing feature importance Residuals - Difference between predicted and actual values R-squared - Measure of model fit (0 to 1)

# APPROACH :

1.Data Collection & Preprocessing
2.Feature Extraction/Selection
Observation:
some features are related
P = V × I

Where:
P is Active Power (Global_active_power)
V is Voltage
I is Current/Intensity (Global_intensity)
Therefore, if we know the active power and voltage, we can directly calculate the intensity. (Note: If we didn't know that we could analyze relationships - e.g. by creating a correlation matrices)

> Derive time-based features to capture usage patterns: hour, day, month
> Use sub_metering readings for areas/appliances

3.Model Selection & Evaluation
> Regression for real valued target
> Compare muliple models


# WHY DO WE SCALE? -

Scaling features is crucial in regression analysis

The StandardScaler performs standardization by:

Computing mean (μ) and standard deviation (σ) for each feature

Transforming each value using: z = (x - μ) / σ

Without scaling, the larger-range features would inappropriately dominate the model's learning process.

(1.) Equal Feature Influence

> Without scaling, features with larger numeric ranges (like household income: $20,000-200,000) would dominate features with smaller ranges (like number of rooms: 1-10)
> Scaling ensures each feature contributes proportionally to the model based on its actual importance, not its numerical magnitude

(2.) Algorithm Performance

> Many algorithms (especially gradient-based ones) converge faster with scaled features
> Linear regression is less affected, but algorithms like gradient descent work better with scaled features
> Neural networks almost always require scaled features for proper convergence

(3.) Mathematical Stability

> Prevents computational issues with very large or very small numbers
> Reduces the chance of overflow/underflow errors
> Helps avoid numerical instability in optimization calculations

(4.)Model Interpretability

> Makes feature importance comparisons more meaningful
> Coefficients become more directly comparable
> Easier to understand the relative impact of each feature
