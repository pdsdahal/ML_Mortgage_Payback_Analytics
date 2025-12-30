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