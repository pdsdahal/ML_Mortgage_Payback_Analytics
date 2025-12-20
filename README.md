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








