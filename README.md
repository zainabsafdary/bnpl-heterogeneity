# bnpl-heterogeneity
Analysis of differences in Buy Now, Pay Later use and repayment behavior among 2,004 BNPL users.

I used the Federal Reserve's 2025 Survey of Household Economics and Decisionmaking (SHED) to identify different types of BNPL borrowers. The analysis:

- groups users based on their financial characteristics using latent class analysis
- identifies three distinct types of BNPL borrowers
- compares late-payment rates across the three groups

Constrained borrowers represented 37% of BNPL users and paid late 40% of the time, compared with 10% among cushioned users with savings, who represented 20% of users.

The analysis was completed in **Python and SQL**, using `pandas`, `NumPy`, `DuckDB`, a custom EM algorithm, and `pytest`.

The full analysis and visualization code is available in the project repository.

Data: Federal Reserve, [2025 Survey of Household Economics and Decisionmaking (SHED)](https://www.federalreserve.gov/consumerscommunities/shed.htm).

## Author

**Zainab Safdary** [Personal Website](https://zainabsafdary.github.io/)
