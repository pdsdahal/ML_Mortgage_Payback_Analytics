# working directory
#getwd()
#setwd("/Users/sdahal/GitRepos/ML_Mortgage_Payback_Analytics")
library(ggplot2)
library(dplyr)
library(psych)
library(corrplot)
library(randomForest)
library(splitTools) # for splitting datasets
library(caret)

# load the dataset
mortgage_payback_data <- read.csv("dataset/Mortgage_Payback.csv")
#View(mortgage_payback_data)

###################################################### 3.1. Data Exploration
# structure
str(mortgage_payback_data)

################# 3.1.1. Variable Definitions
# Summary statistics
#summary(mortgage_payback_data)

################# 3.1.2. Numerical Data
# let define feature groups
numerical_features <- c('id', 'time', 'orig_time', 'first_time', 'mat_time', 'FICO_orig_time',
                        'balance_time', 'LTV_time', 'interest_rate_time', 'hpi_time', 'gdp_time', 'uer_time',
                        'balance_orig_time', 'LTV_orig_time', 'Interest_Rate_orig_time', 'hpi_orig_time')

categorical_features <- c('REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time',
                          'investor_orig_time', 'default_time', 'payoff_time', 'status_time')
# extract the data
numerical_data <- mortgage_payback_data[numerical_features]
#View(numerical_data)
summary(numerical_data)

################# 3.1.3. Data Quality Issues
# Condition 1: Origination happens after first observation
cond1 <- mortgage_payback_data$orig_time > mortgage_payback_data$first_time

# Condition 2: First observation happens after maturity
cond2 <- mortgage_payback_data$first_time > mortgage_payback_data$mat_time

# Condition 3: Observation happens before origination
cond3 <- mortgage_payback_data$time < mortgage_payback_data$orig_time

# Condition 4: Observation happens after maturity
cond4 <- mortgage_payback_data$time > mortgage_payback_data$mat_time

# combine all conditions
time_issues_data <- data.frame(
  Condition = c("orig_time > first_time",
                "first_time > mat_time",
                "time < orig_time",
                "time > mat_time"),
  Count = c(sum(cond1),
            sum(cond2),
            sum(cond3),
            sum(cond4))
)
print(time_issues_data)

# Remove problematic records where orig_time > first_time
mortgage_payback_data <- mortgage_payback_data %>% filter(!(orig_time > first_time))
cat("Remaining observations in dataset:", nrow(mortgage_payback_data))

# ensuring no invalid records remain
remaining_issues <- sum(mortgage_payback_data$orig_time > mortgage_payback_data$first_time)
cat("Remaining problematic records:", remaining_issues)

################# 3.1.4. Analysis of Loan Duration and Balance
loan_duration_analysis <- mortgage_payback_data

# Loan duration analysis
loan_duration <- loan_duration_analysis %>%
  group_by(id) %>%
  summarise(
    loan_duration = max(mat_time) - min(orig_time),
    loan_age = max(time) - min(orig_time),
    observation_period = max(time) - min(time) + 1,
    avg_balance = mean(balance_time),
    balance_reduction = (first(balance_time) - last(balance_time)) / first(balance_time),
    ever_default = max(default_time)
  )
summary(loan_duration$loan_duration)
summary(loan_duration$loan_age)
summary(loan_duration$balance_reduction)

hist(loan_duration$loan_duration,
     main = "Distribution of Loan Duration",
     xlab = "Loan Duration (Months)",
     col = "skyblue", border = "white")

# Histogram of Loan Age
hist(loan_duration$loan_age,
     main = "Distribution of Loan Age",
     xlab = "Loan Age (Months)",
     col = "lightgreen", border = "white")

# main---- get last observations for each id
last_observations <- mortgage_payback_data %>%
  arrange(id, time) %>%
  group_by(id) %>%
  slice_tail(n = 1) %>%
  ungroup()
#View(last_observations)

################# 3.1.5. Analysis of FICO Distribution
hist(last_observations$FICO_orig_time,
     main = "Distribution of Borrower Credit Scores at Origination",
     xlab = "FICO Score",
     ylab = "Number of Borrowers",
     col = "lightgreen",
     border = "white",
     breaks = 30,
     xlim = c(500, 850),
     las = 1)

################# 3.1.6. Analysis Of Interest Rate Distribution by Loan Status
last_observations_int <- last_observations

# factor for status_time
last_observations_int$status_lab <- factor(last_observations_int$status_time,
                                           levels = c(0,1,2),
                                           labels = c("Non-default/Non-payoff",
                                                      "Default",
                                                      "Payoff"))
# Rate orig by status
boxplot(Interest_Rate_orig_time ~ status_lab, data = last_observations_int,
        col = "#33a02c", main = "Interest Rate by status_time", xlab = "", ylab = " Interest Rate (%)", las = 2)

################# 3.1.7. Categorical Data
# Select categorical variables
categorical_data <- last_observations[, categorical_features]

# Build frequency table
freq_list <- list()

for (var in categorical_features) {
  counts <- table(categorical_data[[var]], useNA = "ifany")
  total <- sum(counts)
  # Ensure 0, 1, 2 columns exist
  result <- c("0" = "0 (0%)", "1" = "0 (0%)", "2" = "0 (0%)")
  for (val in names(counts)) {
    if (val %in% c("0", "1", "2")) {
      percent <- round(counts[val] / total * 100, 2)
      result[val] <- paste0(counts[val], " (", percent, "%)")
    }
  }
  freq_list[[var]] <- result
}

# Convert to data frame
freq_table <- as.data.frame(do.call(rbind, freq_list))
freq_table$Variable <- categorical_features
freq_table <- freq_table[, c("Variable", "0", "1", "2")]
# Remove row names to avoid duplication
rownames(freq_table) <- NULL
# Display
print(freq_table)

# status_time bar plot visualization
barplot(table(last_observations$status_time),
        main = "Distribution of Loan Status",
        xlab = "Loan Status Code",
        ylab = "Number of Borrowers",
        col = "lightblue",
        names.arg = c("Active (0)", "Default (1)", "Payoff (2)"),
        las = 1)
table(last_observations$payoff_time)

payOffStatusData <- last_observations
payOffStatusData <- payOffStatusData %>% filter(default_time == 0)

# payoff_time bar plot visualization
color_pay_off_status <- c("lightgreen", "lightskyblue")
barplot(table(payOffStatusData$payoff_time),
        main = "Distribution of payoff time",
        xlab = "payoff time Status Code",
        ylab = "Number of Borrowers",
        col = color_pay_off_status,
        names.arg = c("Active (0)",  "Payoff (1)"),
        las = 1)

###################################################### 3.2. Data Preprocessing 
################# 3.2.1. Checking Missing Value
missing_summary <- colSums(is.na(mortgage_payback_data))
data.frame(Missing_Count = missing_summary, 
           Missing_Percent = paste0(round((missing_summary / nrow(mortgage_payback_data)) * 100, 2), "%"))

# grab records with missing LTV_time
ltv_time_missing <- mortgage_payback_data[is.na(mortgage_payback_data$LTV_time), ]
#View(ltv_missing)
table(ltv_time_missing$investor_orig_time)
table(ltv_time_missing$balance_orig_time)
table(ltv_time_missing$status_time)
table(ltv_time_missing$payoff_time)
table(ltv_time_missing$default_time)
#View(ltv_time_missing)

#length(unique(mortgage_payback_data$id))
#################  3.2.2. Handling Missing
missing_ids <- unique(ltv_time_missing$id)
length(missing_ids)
# let me check if those ids have LTV_time missing for all their records
ltv_missing_check <- mortgage_payback_data %>%
  filter(id %in% missing_ids) %>%
  group_by(id) %>%
  summarise(
    total_records = n(),
    missing_count = sum(is.na(LTV_time)),
    all_missing = all(is.na(LTV_time))
  )

# View result
#ltv_missing_check
# Remove records with missing LTV_time
mortgage_payback_data <- mortgage_payback_data[!is.na(mortgage_payback_data$LTV_time), ]

# Verify removal
sum(is.na(mortgage_payback_data$LTV_time))  
#str(mortgage_payback_data)
#length(unique(mortgage_payback_data$id))

#################  3.2.3. Checking Empty Strings
sum(mortgage_payback_data == "", na.rm = TRUE)

#################  3.2.4. checking zeros entire data set except categorical variables 
check_zeros <- function(data, numeric_cols) {
  zeros_summary <- colSums(data[numerical_features] == 0, na.rm = TRUE)
  # summary with percent
  zeros_df <- data.frame(
    Column = names(zeros_summary),
    Zero_Count = zeros_summary,
    Zero_Percent = paste0(round((zeros_summary / nrow(data)) * 100, 2), "%")
  )
  rownames(zeros_df) <- NULL
  return(zeros_df)
}

zeros_df <- check_zeros(mortgage_payback_data, numerical_features)
print(zeros_df)

# grab records with orig_time == 0 
zero_orig_records <- mortgage_payback_data[mortgage_payback_data$orig_time == 0, ]
#View(zero_orig_records)

# grab records with balance_time == 0 
zero_balance_time <- mortgage_payback_data[mortgage_payback_data$balance_time == 0, ]
table(zero_balance_time$payoff_time)

# grab records with LTV_time == 0 
zero_LTV_time <- mortgage_payback_data[mortgage_payback_data$LTV_time == 0, ]
table(zero_LTV_time$payoff_time)

################# 3.2.5. Handling for Zeros 
# records with balance_time == 0 
zero_balance_time_records <- mortgage_payback_data[mortgage_payback_data$balance_time == 0, ]
#View(zero_balance_time_records)

# before the 285 total number records with payoff_time == 1, status_time == 2 and balance_time == 0
sum(mortgage_payback_data$payoff_time == 1 & mortgage_payback_data$status_time == 2 & mortgage_payback_data$balance_time == 0)

#Get ids where last record has balance_time == 0
ids_to_update <- last_observations %>%
  filter(balance_time == 0, payoff_time == 0, status_time == 0) %>%
  pull(id)

# only 37 id needs to update
length(ids_to_update)

# Update only the last observation for those IDs
mortgage_payback_data <- mortgage_payback_data %>%
  arrange(id, time) %>%
  group_by(id) %>%
  mutate(
    is_last = row_number() == n(),
    payoff_time = ifelse(is_last & id %in% ids_to_update, 1, payoff_time),
    status_time = ifelse(is_last & id %in% ids_to_update, 2, status_time)
  ) %>%
  ungroup() %>%
  dplyr::select(-is_last)

# verify update on balance_time 285 + 37 = 322
sum(mortgage_payback_data$payoff_time == 1 & mortgage_payback_data$status_time == 2 & mortgage_payback_data$balance_time == 0)

################# 3.2.6. Outlier Detection
# Function to calculate outlier count using IQR
find_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  sum(x < (q1 - 1.5*iqr) | x > (q3 + 1.5*iqr), na.rm = TRUE)
}

# apply function to all numeric columns
outlier_counts <- sapply(mortgage_payback_data[numerical_features], find_outliers)

# create summary table without duplicating column names
outliers_df <- data.frame(
  Variable = names(outlier_counts),
  Outlier_Count = as.numeric(outlier_counts),
  Min_Value = round(sapply(mortgage_payback_data[numerical_features], function(x) min(x, na.rm = TRUE)), 2),
  Max_Value = round(sapply(mortgage_payback_data[numerical_features], function(x) max(x, na.rm = TRUE)), 2),
  Percentage = paste0(round(as.numeric(outlier_counts) / nrow(mortgage_payback_data) * 100, 2), "%"),
  row.names = NULL
)
print(outliers_df)

# Predictor Analysis and Relevancy
# 4.3.1. Correlation Analysis for Numeric Predictors
mortage_aggregate <- mortgage_payback_data %>%
  group_by(id) %>%
  summarise(
    # Origination (first)
    FICO = first(FICO_orig_time),
    LTV_orig = first(LTV_orig_time),
    rate_orig = first(Interest_Rate_orig_time),
    balance_orig = first(balance_orig_time),
    hpi_orig = first(hpi_orig_time),
    mat_time = first(mat_time),
    investor_orig_time = first(investor_orig_time),
    REtype_SF_orig_time = first(REtype_SF_orig_time),
    REtype_PU_orig_time = first(REtype_PU_orig_time),
    REtype_CO_orig_time = first(REtype_CO_orig_time),
    
    # Final state (last)
    balance_final = last(balance_time),
    LTV_final = last(LTV_time),
    rate_final = last(interest_rate_time),
    hpi_final = last(hpi_time),
    gdp_final = last(gdp_time),
    uer_final = last(uer_time),
    
    # Time span
    loan_age_months = max(time) - min(time),
    
    default_time   = last(default_time),
    status_time    = last(status_time),
    #target 
    payoff_time    = last(payoff_time),
    .groups = "drop"
  )
# let me track dynamic features
mortage_aggregate <- mortage_aggregate %>%
  mutate(
    balance_reduction = (balance_orig - balance_final) / balance_orig,
    LTV_change = LTV_final - LTV_orig,
    rate_change = rate_final - rate_orig,
    hpi_appreciation = (hpi_final - hpi_orig) / hpi_orig
  )

#filter
mortage_aggregate_final <- mortage_aggregate %>%
  filter(default_time == 1 | payoff_time == 1)

# Data Aggregation Strategy
#View(mortgage_payback_data)
borrower_id_2 <- mortgage_payback_data %>% filter(id == 2)

borrower_agg_id_2 <- mortage_aggregate_final %>% filter(id == 2)

#View(borrower_id_2)
#View(borrower_agg_id_2)
#View(mortage_aggregate_final)
final_numeric_features <- c(
  "FICO", "LTV_orig", "rate_orig", "balance_orig", "mat_time", "hpi_orig",
  "balance_final", "LTV_final", "rate_final", "hpi_final", "gdp_final", "uer_final",
  "loan_age_months", "balance_reduction", "LTV_change", "rate_change", "hpi_appreciation"
)

mortgage_payback_borrower_num <- mortage_aggregate_final[, final_numeric_features]
cor_matrix <- cor(mortgage_payback_borrower_num, use = "complete.obs")
cor_matrix

options(repr.plot.width = 16, repr.plot.height = 14)  
par(mfrow = c(1, 1), mar = c(2, 2, 2, 2))

# Increase text size
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         order = "hclust",
         tl.cex = 0.7,
         tl.col = "black",
         addCoef.col = "black",
         number.cex = 0.6,
         mar = c(0,0,2,0),
         col = colorRampPalette(c("blue", "white", "red"))(200))

#View(mortage_aggregate_final)
#4.3.3. Categorical Predictors Relevancy
mortgage_payback_data_categorical <- mortage_aggregate_final
categorical_features <- c('REtype_CO_orig_time', 'REtype_PU_orig_time', 'REtype_SF_orig_time', 'investor_orig_time', 'payoff_time')

# Ensure all categorical variables are factors
mortgage_payback_data_categorical[categorical_features] <- 
  lapply(mortgage_payback_data_categorical[categorical_features], as.factor)

# Chi-squared test for REtype_CO_orig_time and payoff_time
co_orig_chi <- chisq.test(mortgage_payback_data_categorical$REtype_CO_orig_time, mortgage_payback_data_categorical$payoff_time)
print("Chi-squared test for REtype_CO_orig_time and payoff_time:")
print(co_orig_chi)

# Chi-squared test for REtype_PU_orig_time and payoff_time
pu_orig_chi <- chisq.test(mortgage_payback_data_categorical$REtype_PU_orig_time, mortgage_payback_data_categorical$payoff_time)
print("Chi-squared test for REtype_PU_orig_time and payoff_time:")
print(pu_orig_chi)

# Chi-squared test for REtype_SF_orig_time and payoff_time
sf_orig_chi <- chisq.test(mortgage_payback_data_categorical$REtype_SF_orig_time, mortgage_payback_data_categorical$payoff_time)
print("Chi-squared test for REtype_SF_orig_time and payoff_time:")
print(sf_orig_chi)

# Chi-squared test for investor_orig_time and payoff_time
inv_orig_chi <- chisq.test(mortgage_payback_data_categorical$investor_orig_time, mortgage_payback_data_categorical$payoff_time)
print("Chi-squared test for investor_orig_time and payoff_time:")
print(inv_orig_chi)

#5.2. Data Transformation
##########Feature Importance from Random Forest : 
## Based on correlation matrix removed features are : 
remove_vars <- c("id","rate_final","balance_final","LTV_final","rate_orig","hpi_final","hpi_orig","default_time","status_time")
mortage_aggregate_imp <- mortage_aggregate_final
mortage_aggregate_imp <- mortage_aggregate_imp[, !(names(mortage_aggregate_imp) %in% remove_vars)]
mortage_aggregate_imp$investor_orig_time <- as.factor(mortage_aggregate_imp$investor_orig_time)
mortage_aggregate_imp$REtype_SF_orig_time <- as.factor(mortage_aggregate_imp$REtype_SF_orig_time)
mortage_aggregate_imp$REtype_PU_orig_time      <- as.factor(mortage_aggregate_imp$REtype_PU_orig_time)
mortage_aggregate_imp$REtype_CO_orig_time             <- as.factor(mortage_aggregate_imp$REtype_CO_orig_time)
mortage_aggregate_imp$payoff_time             <- as.factor(mortage_aggregate_imp$payoff_time)

#str(mortage_aggregate_imp)
# Train Random Forest model
set.seed(2025)
rf_model <- randomForest(payoff_time ~ ., 
                         data = mortage_aggregate_imp, 
                         importance = TRUE)
rf_importance <- randomForest::importance(rf_model)
# Convert to data frame
rf_importance_df <- data.frame(
  Variable = rownames(rf_importance),
  MeanDecreaseAccuracy = rf_importance[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini     = rf_importance[, "MeanDecreaseGini"]
)
# Sort by MeanDecreaseAccuracy
rf_importance_df <- rf_importance_df[order(rf_importance_df$MeanDecreaseAccuracy, decreasing = TRUE), ]
rownames(rf_importance_df) <- NULL
print(rf_importance_df)

# Final feature selected Based on correlation matrix, random forest 
remove_features <- c("id","rate_final","balance_final","LTV_final","rate_orig","hpi_final","hpi_orig","REtype_SF_orig_time","REtype_CO_orig_time","REtype_PU_orig_time","default_time","status_time")
mortage_data_class <- mortage_aggregate_final
mortage_data_class <- mortage_data_class[, !(names(mortage_data_class) %in% remove_features)]
#View(mortage_data_class)
#table(mortage_data_class$payoff_time)

###############Data Partition 
strata_factor <- interaction(
  mortage_data_class$payoff_time,
  drop = TRUE
)
# perform a stratified 60/20/20 split
set.seed(2025)
splits <- partition(
  y = strata_factor, 
  p = c(train = 0.6, val = 0.2, test = 0.2),
  type = "stratified"
)
# Extract partitions
train_data <- mortage_data_class[splits$train, ]
val_data   <- mortage_data_class[splits$val, ]
test_data  <- mortage_data_class[splits$test, ]

###################### let create summary
# Partition sizes
train_n <- nrow(train_data)
val_n   <- nrow(val_data)
test_n  <- nrow(test_data)

# Total records
total_n <- nrow(mortage_data_class)

# Create summary table
partition_summary <- data.frame(
  Partition   = c("Training", "Validation", "Test"),
  Records     = c(train_n, val_n, test_n),
  Percentage  = round(c(train_n, val_n, test_n) / total_n * 100, 2)
)

print(partition_summary)
#Verify distributions
round(prop.table(table(train_data$investor_orig_time)) * 100, 2)
round(prop.table(table(val_data$investor_orig_time)) * 100, 2)
round(prop.table(table(test_data$investor_orig_time)) * 100, 2)

round(prop.table(table(train_data$payoff_time)) * 100, 2)
round(prop.table(table(val_data$payoff_time)) * 100, 2)
round(prop.table(table(test_data$payoff_time)) * 100, 2)

# Check overlap between splits
length(intersect(splits$train, splits$test))
length(intersect(splits$train, splits$val)) 
length(intersect(splits$val, splits$test))

# Check coverage
length(unique(c(splits$train, splits$val, splits$test))) == nrow(mortage_data_class)

# Algorithm Selected 
# 7.1. Classification Algorithms
# 7.1.1. Logistic Regression
# 7.1.2. Random Forest 

# 7.2. Clustering Algorithms
# 7.2.1. K-Means

cat("Remaining observations in dataset:", nrow(mortage_data_class))

#  factors common method
factorize_predictors <- function(df) {
  df$investor_orig_time <- as.factor(df$investor_orig_time)
  return(df)
}
# Apply function
X_train_cat <- factorize_predictors(train_data[, !names(train_data) %in% "payoff_time"])
X_val_cat   <- factorize_predictors(val_data[, !names(val_data) %in% "payoff_time"])
X_test_cat  <- factorize_predictors(test_data[, !names(test_data) %in% "payoff_time"])

# Targets as factors (with explicit levels for consistency)
y_train_cat <- factor(train_data$payoff_time, levels = c(0, 1))
y_val_cat   <- factor(val_data$payoff_time, levels = c(0, 1))
y_test_cat  <- factor(test_data$payoff_time, levels = c(0, 1))

##################################### Logistic Regression
X_train_cat_lr <- X_train_cat
X_val_cat_lr   <- X_val_cat
X_test_cat_lr  <- X_test_cat
library(glmnet)
library(pROC)

# Fit default Logistic Regression model
lr_model_default <- glm(payoff_time ~ ., 
                        data = cbind(X_train_cat_lr, payoff_time = y_train_cat),
                        family = binomial)
# Check convergence and print summary
cat("Logistic Regression Default Model Summary:\n")
summary(lr_model_default)

# Evaluation function
evaluate_lr <- function(model, X, y, threshold = 0.5) {
  pred_prob <- predict(model, newdata = X, type = "response")
  pred_class <- ifelse(pred_prob > threshold, 1, 0)
  cm <- confusionMatrix(factor(pred_class), factor(y), positive = "1")
  roc_obj <- roc(y, pred_prob, quiet = TRUE)
  
  metrics <- list(
    Accuracy = cm$overall["Accuracy"],
    Sensitivity = cm$byClass["Sensitivity"],
    Specificity = cm$byClass["Specificity"],
    Precision = cm$byClass["Precision"],
    F1 = cm$byClass["F1"]
  )
  return(metrics)
}

# Evaluate default model
lr_val_metrics <- evaluate_lr(lr_model_default, X_val_cat_lr, y_val_cat)
cat("Logistic Regression Default Validation Metrics:\n")
print(sapply(lr_val_metrics, round, 3)) 

lr_test_metrics <- evaluate_lr(lr_model_default, X_test_cat_lr, y_test_cat)
cat("Logistic Regression Default Test Metrics:\n")
print(sapply(lr_test_metrics, round, 3))

##################################### Tune Logistic Regression
library(MASS)
library(caret) 

# Stepwise selection (both directions) on training data
lr_model_step <- stepAIC(lr_model_default, direction = "both", trace = FALSE)
cat("Logistic Regression Stepwise Model Summary:\n")
summary(lr_model_step)

#View(X_train_cat_lr)
# Remove balance_orig (weak feature)
X_train_final_lr <- X_train_cat_lr %>% dplyr::select(-balance_orig, -uer_final, -rate_change)
X_val_final_lr   <- X_val_cat_lr %>% dplyr::select(-balance_orig, -uer_final, -rate_change)
X_test_final_lr  <- X_test_cat_lr %>% dplyr::select(-balance_orig, -uer_final, -rate_change)

# ReFit logistic regression model on selected features
lr_model_final <- glm(payoff_time ~ ., 
                      data = cbind(X_train_final_lr, payoff_time = y_train_cat),
                      family = binomial)

# Run stepwise AIC again
lr_model_step_final <- stepAIC(lr_model_final, direction = "both", trace = FALSE)

# Review final model summary
cat("Final Stepwise Logistic Regression Summary:\n")
summary(lr_model_step_final)

thresholds <- seq(0.4, 0.6, by = 0.01)
# Evaluate metrics for each threshold
results_list <- lapply(thresholds, function(t) {
  evaluate_lr(lr_model_step_final, X_val_final_lr, y_val_cat, threshold = t)
})

# Convert list to data.frame
results <- do.call(rbind, results_list)
results <- data.frame(Threshold = thresholds, results)

# Now find best threshold for F1
best_f1_idx <- which.max(results$F1)
best_threshold <- results$Threshold[best_f1_idx]
cat("Best threshold (F1-optimal) on validation set:", best_threshold, "\n")

##################### Evaluate Final Model
# Validation set metrics with tuned threshold
#lr_val_metrics_tuned <- evaluate_lr(lr_model_step_final, X_val_final_lr, y_val_cat, threshold = best_threshold)
#cat("Validation Metrics (Stepwise Model + Tuned Threshold):\n")
#print(sapply(lr_val_metrics_tuned, round, 3))

# Test set metrics with tuned threshold
lr_test_metrics_tuned <- evaluate_lr(lr_model_step_final, X_test_final_lr, y_test_cat, threshold = best_threshold)
cat("Test Metrics (Stepwise Model + Tuned Threshold):\n")
print(sapply(lr_test_metrics_tuned, round, 3))

##################################### Random Forest Classification
X_train_rf_cls <- X_train_cat
X_val_rf_cls   <- X_val_cat
X_test_rf_cls  <- X_test_cat

# Random Forest classification
set.seed(123)
rf_model_cls <- randomForest(x=X_train_rf_cls,
                             y=y_train_cat, 
                             ntree = 100,
                             mtry = 4,
                             nodesize = 3,
                             importance = TRUE)
print(rf_model_cls)

# eval random forest on validation set
rf_cls_val_pred <- predict(rf_model_cls, X_val_rf_cls)
rf_cls_val_metrics <- confusionMatrix(rf_cls_val_pred, y_val_cat,positive = "1")
precision_val <- rf_cls_val_metrics$byClass["Precision"]
f1_val <- rf_cls_val_metrics$byClass["F1"]
print(rf_cls_val_metrics)
print(precision_val)
print(f1_val)

# eval random forest on test set
rf_cls_test_pred <- predict(rf_model_cls, X_test_rf_cls)
rf_cls_test_metrics <- confusionMatrix(rf_cls_test_pred, y_test_cat,positive = "1")
precision_test <- rf_cls_test_metrics$byClass["Precision"] 
f1_test <- rf_cls_test_metrics$byClass["F1"]
print(rf_cls_test_metrics)
print(precision_test)
print(f1_test)

##################################### Tuned Random Forest Classsification
library(caret)
library(ranger)

# ensuring factor levels 
y_train_cat <- factor(y_train_cat)
y_val_cat   <- factor(y_val_cat)
y_test_cat  <- factor(y_test_cat)

# convert levels like 0/1 to valid names X0/ X1
levels(y_train_cat) <- make.names(levels(y_train_cat))
levels(y_val_cat)   <- make.names(levels(y_val_cat))
levels(y_test_cat)  <- make.names(levels(y_test_cat))

# combine features and target
train_data_rf <- cbind(X_train_rf_cls, Purchase = y_train_cat)

# Number of predictors
p1 <- ncol(train_data_rf) - 1
# hyperparameter grid
tuneGridCs <- expand.grid(
  mtry = c(floor(sqrt(p1)), floor(p1/3)),
  splitrule = c("gini"),
  min.node.size = c(5, 7)
)

# Number of trees to try separately
ntree_values <- c(400, 500)

# setup train control
ctrl <- trainControl(
  method = "cv",
  number = 5,       
  classProbs = TRUE 
)
# Loop over ntree values

best_model <- NULL
best_sensitivity <- -Inf
best_params <- list()

set.seed(123)
for (ntree in ntree_values) {
  
  # Train Random Forest with caret
  rf_model <- train(
    Purchase ~ .,
    data = train_data_rf,
    method = "ranger",
    tuneGrid = tuneGridCs,         
    num.trees = ntree,
    importance = "impurity",
    trControl = ctrl,
    metric = "Accuracy"
  )
  
  # Evaluate on validation set
  preds_val <- predict(rf_model, X_val_rf_cls)
  cm_val <- confusionMatrix(preds_val, y_val_cat, positive = "X1")
  sensitivity_val <- cm_val$byClass["Sensitivity"]
  
  # Keep best model based on Sensitivity
  if (sensitivity_val > best_sensitivity) {
    best_sensitivity <- sensitivity_val
    best_model <- rf_model
    best_params <- list(ntree = ntree, params = rf_model$bestTune)
  }
}

cat("Best parameters based on validation set :")
print(best_params)
print(best_sensitivity)

# evaluation on Validation set
#rf_tuned_val_pred_cs <- predict(best_model, X_val_rf_cls)
#rf_tuned_val_metrics_cs <- confusionMatrix(rf_tuned_val_pred_cs, y_val_cat, positive = "X1")
#f1_tuned_val <- rf_tuned_val_metrics_cs$byClass["F1"]
#precision_tuned_val <- rf_tuned_val_metrics_cs$byClass["Precision"] 
#print(rf_tuned_val_metrics_cs)
#print(f1_tuned_val)
#print(precision_tuned_val)

# evaluation on Test set
rf_tuned_test_pred_cs <- predict(best_model, X_test_rf_cls)
rf_tuned_test_metrics_cs <- confusionMatrix(rf_tuned_test_pred_cs, y_test_cat, positive = "X1")
f1_tuned_test <- rf_tuned_test_metrics_cs$byClass["F1"]
precision_tuned_test <- rf_tuned_test_metrics_cs$byClass["Precision"] 
print(rf_tuned_test_metrics_cs)
print(f1_tuned_test)
print(precision_tuned_test)