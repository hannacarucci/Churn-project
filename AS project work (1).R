
# Import necessary libraries


library(ggcorrplot) # to have a graphical visualization of the correlation matrix
library(fastDummies) # to transform categorical variables into dummy ones
library(readr) # to read the csv file containing the data
library(tidyverse) 
library(datarium)
library(ggplot2) # to make some plots
library(rsm)
library(MASS)
library(olsrr) 
library(car)
library(ISLR)
library(corrplot)
library(caret) # to perform train-test split


# Import the dataset to R

WA_Telco_Customer_Churn <- read_delim("WA_Telco_Customer_Churn.csv", 
                                      delim = ";", escape_double = FALSE, trim_ws = TRUE)
summary(WA_Telco_Customer_Churn)

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# PRELIMINARY ANALYSIS
# 1. Control for missing data in our dataset

WA_Telco_Customer_Churn <- na.omit(WA_Telco_Customer_Churn)

# Look at the number of NA values in the new dataset to be sure they have been correctly removed

sum(is.na(WA_Telco_Customer_Churn))

# Remove the customer ID and churn variables since they have no role in our regression analysis

data <- WA_Telco_Customer_Churn[,-1]
data <- data[,-20]

# Quickly create dummy (binary) columns from character and factor type columns in the input data

new_data <- fastDummies::dummy_cols(data)
new_data <- new_data[,-1:-18]
new_data$tenure <- data$tenure 
# new_data$MonthlyCharges <- NULL
view(new_data)

# Remove one of each new category that has been created for the categorical variables and save
# the results in a new dataset

z <- c(-3,-5,-7,-9,-12,-15,-18,-21,-24,-27,-30,-33,-36,-38,-42)
new_data_nocol <- new_data[,z]
view(new_data_nocol) 

# ------------------------------------------------------------------------------------------------------
# 2. Analysis of correlation between the predictors

cor <- as.matrix(round(cor(new_data_nocol[,-1]),3))
ggcorrplot(cor, type = "lower", insig = "blank")

# From the correlation matrix we see that the categorical variables with value No_internet_service
# are perfectly correlated.Also, if there is no phone service the variable multiple line
# is perfectly correlated with the PhoneService_No. We remove these variables.

n <- c(-7,-11,-13,-15,-17,-19,-20)
new_data_nocol <- new_data_nocol[,n] 

# Plot the new correlation matrix

cor <- as.matrix(round(cor(new_data_nocol[,-1]),3))
ggcorrplot(cor, type = "lower", insig = "blank")

# The matrix shows only mild correlations left in our predictors.
# These are now ready to be used in our regression analysis.

# Create interaction variables for later use.



# ------------------------------------------------------------------------------------------------------
# 3. Analysis of the target variable distribution

target <- as.vector(new_data$TotalCharges)
hist(target, breaks = 50)

# Y's distribution clearly not normal

# Randomly chose a sub-sample of y to perform the Shapiro test

set.seed(123)
y_random <- sample(x = target, size = 5000)
hist(y_random)
# Given the histogram of y, the values of lambda (in the box-cox formula) to
# better approximate the distribution of the transformed y to a normal one are comprised between 0 and 0.5.
# The code below performs a series of Shapiro tests on the aforementioned range of lambda values

for (i in seq(0,0.5,0.1)) {
  if (i != 0) {
    y_test <- (y_random^(i)-1)/i
  }
  else {
    y_test <- log(y_random)
  }
  print(shapiro.test(y_test))
  print(i)
}

# The best fitting value is 0.3
BoxCox <- function(x,lambda) {(x^(lambda)-1)/lambda}
hist(BoxCox(y_random,0.3), breaks = 50)
new_data_nocol$TotalCharges <- BoxCox(new_data_nocol$TotalCharges,0.3)
hist(new_data_nocol$TotalCharges, breaks = 50)

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# DATA SPLITTING
# Divide data set in train and test data 

## Take 75% of the sample size
smp_size <- floor(0.75 * nrow(new_data_nocol))

## Set the seed to make the partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(new_data_nocol)), size = smp_size)

train <- new_data_nocol[train_ind, ]
test <- new_data_nocol[-train_ind, ]

y_train <- as.vector(train[,1]) 
x_train <- as.vector(train[,21])

# -------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------
# TRAINING
# Try out a simple model first

linear_model = lm(unlist(y_train) ~ unlist(x_train)) 
summary(linear_model)

# The R-squared is a measure of fit that takes the form of a proportion. It measures the proportion of variability 
# in the response variable Y that is explained using X. 
# Here, the obtained R-squared is approximately 0.757. 

# The RSE (or model sigma), corresponding to the prediction error, represents roughly 
# the average difference between the observed outcome values and the predicted values 
# by the model. The lower the RSE the best the model fits to our data.
# Dividing the RSE by the average value of the outcome variable will give you the 
# prediction error rate, which should be as small as possible. 
# In our example, using only tenure predictor variable, the RSE = 5.752, meaning that 
# the observed sales values deviate from the predicted values by approximately 5.752 units in average.
# This corresponds to an error rate of 5.752/mean(train.data$tenure) = 2.11/16.77 = 17.7%, which is low.


# We try to use the step AIC to see if we can find something better.

x_train <- as.vector(train[,2:21])
linear_model2 = lm(unlist(y_train) ~., data = x_train)
linear_model_AIC = stepAIC(linear_model2, direction = "both")
summary(linear_model_AIC)
linear_model_AIC$anova # display the procedure
linear_model_AIC$call # less predictors
vif(linear_model_AIC) # No variables with VIF>10. 

# The results are definitely better. 
 


# External validity of the model:
# Are there non-linear relationships? 
non_linear_model <- lm(unlist(y_train) ~ x_train$tenure)
plot(x_train$tenure, unlist(y_train), pch = 16,
     xlab = "tenure",
     ylab = "TotalCharges")
abline(non_linear_model, lwd = 3, lty = 2, col = 2)

# The red line represents the linear regression fit. 
# There is a pronounced relationship between mpg and horsepower, but it seems clear that 
# this relationship is  non-linear: the data suggest a curved relationship. 

# A simple approach for incorporating non-linear associations in a linear model is to
# include transformed versions of the predictors in the model.

plot(non_linear_model, which = 1) # Inspect the residuals VS fitted values...

# In the MLR, since there are a lot of predictors, we plot the residuals versus
# the predicted (or fitted) values. Ideally, the residual plot will show no
# fitted discernible pattern. 
# The presence of a pattern may indicate a problem with some aspect of the linear model.

linear_model_AIC2 = update(linear_model_AIC, ~.+I(tenure^(1/2)))
summary(linear_model_AIC2)

# The quadratic fit appears to be substantially better 
# than the fit obtained when just the
# linear term is included. The R2 of the quadratic fit is 0.940, compared to
# 0.984 for the linear fit, and the p-value for the quadratic term
# is highly significant.

# The approach that we have just described for extending the linear model
# to accommodate non-linear relationships is known as polynomial regression,
# since we have included polynomial functions of the predictors in the
# regression model

                   
# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# RESIDUAL ANALYSIS
library(ggplot2)
library(olsrr)

ols_plot_resid_qq(linear_model_AIC2) # Building the graph for testing normality assumption 


ols_plot_resid_fit(linear_model_AIC2) # Not so constant variance.
# More generally, the non-random pattern in the residuals indicates that the predictor variables 
# are not capturing some explanatory information that is present into the residuals. 
# The graph could represent several ways in which the model is not explaining all that is possible. 
# Possibilities include:
# 1. A missing variable
# 2. A missing higher-order term of a variable in the model to explain the presence of a curvature
# 3. A missing interaction between terms already in the model
# 4. The presence of correlation between errors and predictors 

ols_plot_resid_hist(linear_model_AIC2) # Residual histogram, this will show residual distribution 




# --------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------
# TESTING

y_test <- unlist(test[1:1758,1])

prediction2 <- predict(linear_model_AIC, test[])
R_squared <- 1-sum((y_test-prediction2)^2)/sum((y_test-mean(y_test))^2)*(length(y_test)-1)/(length(y_test)-21)
R_squared

prediction3 <- predict(linear_model_AIC2, test[])
R_squared2 <- 1-sum((y_test-prediction3)^2)/sum((y_test-mean(y_test))^2)*(length(y_test)-1)/(length(y_test)-21)
R_squared2


