# ML_Mortgage_Payback_Analytics
The purpose of the Mortgage Payback Analytical is to build a classification model that predicts the probability of a mortgage being paid off and segment borrowers into actionable groupings for improved risk management. 

# Introduction
The mortgage dataset contains information on 50,000 home loans made in the U.S. over 60 time periods. Every row reflects where the loan began and how it performed over time (e.g., payment made versus payment missed). Some of the loans in this data started before the collection period (loans can be sold between banks and mortgage investors), while others were labeled as paid off or refinanced early. The data is based on a random sample of large groups of mortgages placed into mortgage-backed securities (RMBS) that were provided to International Financial Research for analytical purposes.

# Problem
Mortgage payoffs reduce expected interest income as well as uncertainty in both revenue and portfolio planning. Lenders often lack timely identification of paid-off customers, which limits their ability to retain those customers. 

**Why identify paid-off customers**

Identifying customers who have paid off their loans quickly allows lenders to market other financial services/products to them and keep them involved with the lender so they can reduce the impact of lost future interest income.

## Business Goal
- Find a way to identify which borrowers are most likely to pay off their mortgage.
- Improve lending decisions and optimize portfolio management.

## Analytical Goal

- **Analytical Goal 1**
  - Explore and preprocess the mortgage dataset to understand borrower characteristics and prepare it for predictive modeling.
  - Build a classification model to predict the probability of a mortgage being paid off.

- **Analytical Goal 2**
  - Segment paid-off borrowers into distinct groups based on their loan and economic characteristics.
  - Analyze patterns in each group to understand their behavior and value.

# Ways to Address
- [x] Data Exploration
- [x] Data Preprocessing
- [x] Predictor Analysis and Relevancy
- [x] Data Dimension and Transformation
- [x] Data Partitioning
- [x] Model selection
- [x] Model Fitting, Validation Accuracy, and Test Accuracy
- [x] Model Evaluation
- [x] Recommendations

# Insights
- Loans that went into default had the highest rates, suggesting that riskier borrowers paid more.
- Loans that were fully paid had the lowest rates, suggesting that safer borrowers received better terms.
- The majority of loans are not condominiums, 93.5% of observations fall under 0.
- Planned urban developments are uncommon but slightly more frequent than condominiums.
- Single-family homes are the dominant type among property types.
- Most borrowers are owners, indicating the smallest portion of loans to investors.

### Loan Status Analysis
- 69.69 % of loans fall under observations of not being in default (0), 30.31% of loans are observed in default (1).
- 46.83 % of the loans are not paid off (0), and 53.17% are paid off (1).
- There are observations for 16.52% of the loans being active (0), 30.31 % in default (1), with 53.17 % paid off (2).

### time
- Time ranges from 1 to 60, with a mean of 35.8. Most observation times are between 27 and 44.
- Observations span 60 periods, centered around period 34, suggesting a balanced temporal dataset for analyzing loan performance trends.

### orig_time
- orig_time ranges from -40 to 60, with an average of 20.57. The most origination times is 18 to 25.
- Negative values represent loans that originated prior to the observation period and indicate that censoring may be possible. 

### first_time
- First observation time ranges from 1 to 60, with an average of 24.61. Most first observation times are between 21 and 28.

### mat_time
- Maturity Time ranges from 18 to 229, with an average of 137.2. Most maturity times are between 137 and 145.
- 75% of loans mature within 145 months.

### FICO_orig_time
- 75% of borrowers have scores >=729.
- FICO scores are largely concentrated between 580-800, predominantly in the 680-750 range, suggesting that most borrowers are classified as Good to Very Good credit categories.

### balance_time
- balance_time ranges from 0 to 8,701,859, with an average of 245,965. Most balances are between 102,017 and 337,495.
- A balance_time of 0 is not realistic for an active loan. The balance_time of 0 usually indicates invalid data or missing information.

### LTV_time
- Loan-to-Value Ratio at observation time ranges from 0 to 803.51, with an average of 83.08. Most LTV ratios are between 67.11 and 100.63.(270 records with missing values)

### interest_rate_time
- interest_rate_time ranges from 0 to 37.5, with an average of 6.702. Most interest rates are between 5.65 and 7.875.
  
### hpi_time
- House Price Index at Observation Time ranges from 107.8 to 226.3, with an average of 184.1. Most hpi_time values are between 158.6 and 212.7.

### gdp_time
- gdp_time ranges from -4.147 to 5.132, with an average of 1.381. Most GDP growth rates are between 1.104 and 2.694.
- Negative values show economic decline that may raise default risk. 

### uer_time
- Unemployment Rate ranges from 3.8 to 10, with an average of 6.517. Most unemployment rates are between 4.7 and 8.2.

### balance_orig_time
- balance_orig_time ranges from 0 to 8,000,000, with an average of 256,254. Most origination balances are between 108,000 and 352,000.
- The minimum balance at origination is 0, which indicates a data entry error or extreme outlier.

### LTV_orig_time
- Loan-to-Value Ratio at Origination ranges from 50.1 to 218.5, with an average of 78.98. Most LTV ratios at origination are between 75 and 80.

### Interest_Rate_orig_time
- Interest_Rate_orig_time (Interest Rate at Origination) ranges from 0 to 19.75, with an average of 5.65. Most origination interest rates are between 5 and 7.456.

### hpi_orig_time
- House Price Index at Origination ranges from 75.71 to 226.29, with an average of 198.12. Most hpi_orig_time values are between 179.45 and 222.39.

## Data Quality Issues
### Time Inconsistencies
| Condition | Rule | Description |
|----------|------|-------------|
| 1 | orig_time > first_time | Loan appears in the data before it was created. That is not possible.|
| 2 | first_time > mat_time | The first record is after the loan's maturity date, and of course, that makes no sense.|
| 3 | time < orig_time | The observation occurred before the loan began, and that is not possible.|
| 4 | time > mat_time | The observation occurs after the loan's maturity, and obviously, the loan should no longer exist after that date and time.|

- The first problem (orig_time > first_time) indicates that 45 loans have an origination date that is later than their first observation time, which is not logically possible.
- Instead of trying to fix the 45 inconsistent records out of 622,489 total observations, it was decided to remove them.
- This ensures that all the remaining observations used in the preparation and modeling maintain accurate and realistic time relationships. 

## Handling Missing
### Omission
- LTV_time
  - The LTV_time column is entirely composed of NAs for borrowers with missing loan ID.
  - Since there is no data on these loan IDs, rows associated with them have been removed from the data to facilitate meaningful analyses.
- Original Loan Balance (balance_orig_time)
  - The records show the original loan balance as 0 for all 270 records, indicating that these loans had no original loan balance at the time the loan was originated.
- Loan Status (status_time)
  - 258 records have a loan status of non-default/non-payoff, 5 records are a default, and 7 are payoff status.
- Payoff status (payoff_time)
  - 263 records are non-paid, and 7 records are paid.
- Default status (default_time)
  - 265 records are non-defaulted, and 5 records are defaulted.

**Conclusion**

- These 270 records of loans are inactive or anomalous loans with no origination loan balance that are mostly non-investor and generally neither non-defaulted nor non-paid loans.
- The impacts of the missing values of LTV_time reflect more issues with the quality of the data and rather than valid observations.
- These records will be removed from the dataset to ensure the integrity of the data.

## Empty Strings
- No empty strings were found in the dataset.

## Handling for Zero
- Loans with a zero interest rate will be retained in the dataset, as these loans could reflect actual promotional rates. Loans with a zero-ending balance will also stay in
  the dataset, as these loans are likely fully paid.
- Based on the business discussion, the only record updated with balance_time = 0 was the last record within the loan being paid-off. The record was updated to reflect
  that payoff_time = 1 and status_time = 2 to accurately capture a paid-off loan is completed without modifying any prior record.

## Handling Outliers
- All flagged values were verified as valid values and significant within the mortgage portfolio. The ranges present, particularly in LTV, interest rates, and maturity time
  (mat_time), reflect differences in loan products, differences in borrower types, and cycles in the market, and not data errors.

## Outcome Variables
- Classification Outcome: **payoff_time (1/0)**
- Clustering Outcome: **Clustering does not have a singular outcome variable. Instead, it forms groups based on patterns in variables.**

### Data Aggregation Strategy
The dataset containing mortgage data provides multiple time-stamped observations for each borrower. Each borrower could have monthly or periodic observations of a set of loan attributes, including the monthly loan balance, loan-to-value ratio, mortgage interest rate, and other similar metrics.

To reshape the mortgage data per-borrower to create a one-row per-borrower that summarizes:

- The origination (first observation) information  
- The final state (last observation) information  
- Duration of the loan (amount of time the loan has been occurring).

Origination level variables such as, FICO score, original balance, loan-to-value (LTV), interest rate, housing price index (HPI), and maturity time, were taken from each borrower's first observation. Current-state variables such as balance_time, LTV_time, interest_rate_time, hpi_time, gdp_time, and uer_time were taken from the last observation and renamed with the suffix _final.

- To capture the borrower behavior throughout the life cycle of the loan, four dynamic features were created:

#### Balance reduction
- The percentage of the original balance that was paid down.  
- balance_reduction = (balance_orig − balance_final) / balance_orig

#### LTV change
- The difference between the final and original LTV (negative = equity gain)  
- LTV_change = LTV_final − LTV_orig

#### Rate change
- The difference between the final and original interest rates (possibly indicative of penalties and/or adjustments)  
- rate_change = rate_final − rate_orig

#### HPI appreciation
- Relative change in the local housing price index  
- hpi_appreciation = (hpi_final − hpi_orig) / hpi_orig

## Feature Selection
### Important Predictors
- balance_reduction (137.91)  
- FICO (115.06)  
- balance_orig (83.13)  
- loan_age_months (78.90)  
- LTV_orig (78.26)  
- LTV_change (71.58)
- gdp_final (66.91)  
- mat_time (59.25)  
- hpi_appreciation (50.003)  
- rate_change (47.90)  
- uer_final (43.03)  
- investor_orig_time (30.49)  

The variables REtype_SF_orig_time (5.56), REtype_CO_orig_time (5.84), and REtype_PU_orig_time (2.14) were eliminated from the model as they provide little useful information. They do not assist with predicting loan payoff behavior, and they only contribute noise in the model. Removing them makes the model simpler and keeps the focus on more important borrower and economic factors.

## Data Transformation 
 - Scaling Numerical Variables (KMeans)
 - One Hot Encoding for Categorical Variables

## Data Partitioning
Stratified Splitting
| Training Set | Validation Set | Test Set |
|----------|----------|----------|
| 60%  | 20%  | 20%  |

## Algorithm Selection
- Classification
    - Logistic Regression
    - Random Forest
- Clustering 
    - K-means Clustering with initial k
    - K-means Clustering with best k
 
## Classification Models Evaluation
| Model               | Dataset | Accuracy | Sensitivity | Specificity | Precision | F1-score |
|---------------------|---------|----------|-------------|-------------|-----------|----------|
| Logistic Regression | Test    | 0.773    | 0.888       | 0.572       | 0.785     | 0.833    |
| Random Forest       | Test    | 0.8047   | 0.8747      | 0.6817      | 0.8284    | 0.8509   |

- Random Forest Leads in all four key metrics (Accuracy, Specificity, Precision, and F1-score)

| Metric | k=2 | k=3 |
|-------|-----|-----|
| Overall Mean Silhouette | 0.3802 | 0.3741 |
| **Cluster Distribution**         |            |            |
| Cluster 1 Size | 5318 | 4848 |
| Cluster 2 Size | 21298 | 938 |
| Cluster 3 Size |  | 20830 |
| **Cluster Quality (Silhouette Width)** |        |            |
| Segment 1 Quality | 0.2168661 | 0.3083655 |
| Segment 2 Quality | 0.4209402 | 0.2588319 |
| Segment 3 Quality |  | 0.3946332 |
| **Statistical Distribution**     |            |            |
| Minimum | -0.1627 | -0.171 |
| 1st Quartile | 0.3166 | 0.3283 |
| Median | 0.4172 | 0.4034 |
| 3rd Quartile | 0.4747 | 0.4518 |
| Maximum | 0.5695 | 0.5525 |


## Recommendations
### Targeted Payoff Risk Management

Deploy the Random Forest payoff prediction model to identify the customers with the highest likelihood of paying off their loans, enabling more efficient intervention and portfolio protection.

Implement tiered marketing strategies based on the 3-cluster segmentation:

**Base Segment – Cluster 1 (Financially Stressed Borrowers):**
- Borrowers showing signs of financial strain can be retained by businesses offering new financial products, such as refinancing, personal loans, or credit lines, that provide value and encourage continued engagement.

**Premium Segment – Cluster 2 (Aggressive Equity Builders):**
- High-value borrowers who demonstrate good financial behavior. Ideal for retention, cross-selling, and long-term relationship strengthening.

**Medium Segment – Cluster 3 (Stable Core Borrowers):**
- Reliable and consistent borrowers who benefit from steady engagement and loyalty-focused programs.

### Allocation of Resources

- Invest more in high-value borrowers (Cluster 2) who offer the greatest long-term business opportunity and respond well to targeted retention offers.
- Allocate moderate resources to stable borrowers (Cluster 3) to maintain satisfaction while ensuring operational efficiency.
- Reduce support and spending for High-Risk Borrowers (Cluster 1) and employ minimum tracking and interventions to prevent losses.



