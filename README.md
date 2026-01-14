# MLB Free Agent Salary Prediction using Data Mining Techniques

A comprehensive data mining project predicting MLB free agent contract values (Average Annual Value) using multiple regression and machine learning models with Python, scikit-learn, and baseball analytics data from FanGraphs and pybaseball.

## Overview

This project tackles the challenge of predicting MLB free agent salaries by building and comparing multiple regression models. Using historical player performance statistics and actual contract data from 2022-2024 free agent classes, the analysis develops predictive models to estimate Average Annual Value (AAV) for both position players and pitchers.

## Business Problem

**Challenge**: MLB teams need to evaluate fair market value for free agents to make informed contract decisions
**Impact**: Overpaying or underpaying players affects team competitiveness and long-term financial flexibility
**Solution**: Data-driven salary prediction models using advanced baseball analytics and machine learning

## Dataset

**Source**:
- **pybaseball**: MLB player statistics retrieval
- **FanGraphs**: Advanced baseball metrics and contract data
- **Time Period**: 2022-2024 MLB free agent classes (using prior season statistics)

**Player Categories**:
- **Position Players**: Batters and fielders
- **Pitchers**: Starting and relief pitchers

**Target Variable**:
- `AAV`: Average Annual Value of player contracts (in USD)

**Position Player Features (33 variables)**:
- Service Time, Games (G), At Bats (AB), Plate Appearances (PA)
- Counting Stats: Hits (H), Home Runs (HR), RBIs, Stolen Bases (SB)
- Rate Stats: AVG, OBP, SLG, OPS, ISO, BABIP
- Advanced Metrics: wOBA, xwOBA, wRC+, WAR
- Batted Ball: Line Drive % (LD%), Ground Ball % (GB%), Fly Ball % (FB%)
- Contact Quality: Soft%, Med%, Hard%
- BB% (Walk Rate), K% (Strikeout Rate)

**Pitcher Features**:
- Innings Pitched (IP), Games Started (GS), Strikeouts (K)
- ERA, FIP (Fielding Independent Pitching), xFIP, WHIP
- K/9, BB/9, K/BB ratio
- Ground Ball Rate, Fly Ball Rate
- WAR, Win-Loss Record

## Methodology

### 1. Data Acquisition & Compilation

**Free Agent Data Collection**:
- Extracted 2022, 2023, and 2024 free agent contract information
- Retrieved corresponding player statistics from prior season (2021, 2022, 2023 stats)
- Handled players without contracts (excluded from analysis)
- Separated position players and pitchers into distinct datasets

**Data Merging**:
- Integrated FanGraphs advanced metrics with contract data
- Selected relevant statistical columns for prediction
- Handled missing values through imputation

### 2. Exploratory Data Analysis (EDA)

**Position Player Analysis**:
- AAV distribution analysis (histogram with 10 bins)
- Handedness impact on salary (Left, Right, Switch hitters)
- Seasonal trends in contract values (2022-2024)
- WAR vs. AAV correlation analysis
- Summary statistics for key metrics

**Pitcher Analysis**:
- AAV distribution for pitchers
- Comparison of position player vs. pitcher salary patterns
- Statistical summaries and missing value assessment

**Key Comparisons**:
- Position player vs. pitcher salary distributions
- Seasonal variations in free agent market
- Relationship between performance metrics (WAR) and contract value

### 3. Multiple Linear Regression (MLR)

#### Position Player Models

**Model 1: All Variables**
- Used all 33 predictor variables
- Training/Validation split: 80/20
- Evaluation metrics: R², RMSE
- Feature selection not applied

**Model 2: Forward Selection**
- Stepwise forward feature selection using `mlxtend.SequentialFeatureSelector`
- Scoring: R² with 5-fold cross-validation
- Automatic selection of optimal feature subset
- Execution time: ~23 seconds

**Model 3: Backward Selection**
- Stepwise backward elimination
- Scoring: R² with 5-fold cross-validation
- Started with all features, iteratively removed least important
- Execution time: ~2 seconds

#### Pitcher Models

**Model 1: All Variables**
- Complete pitcher statistics as predictors
- Similar structure to position player model

**Model 2: Forward Selection**
- Feature selection for pitcher-specific metrics
- Execution time: ~12 seconds

**Model 3: Backward Selection**
- Backward elimination for pitchers
- Execution time: ~12 seconds

**Validation Comparison**:
- Bar plots comparing validation RMSE across MLR approaches
- Identified best-performing feature selection method

### 4. Decision Tree Regression

#### Position Player Decision Tree

**Hyperparameter Tuning**:
- Tested max_depth from 1 to 25
- Optimal depth: 4 (based on validation RMSE)
- Random state: 0 for reproducibility

**Model Features**:
- `DecisionTreeRegressor` from scikit-learn
- Visualization: 25x15 inch tree plot
- Feature importance ranking (top 12 displayed)
- Bar plot of variable importance

#### Pitcher Decision Tree

**Hyperparameter Tuning**:
- Tested max_depth from 1 to 19
- Optimal depth: 2 (based on validation RMSE)
- Visualization of tree structure
- Feature importance analysis

**Insights**:
- Line plots showing validation RMSE vs. tree depth
- Identified optimal complexity to avoid overfitting

### 5. K-Nearest Neighbors (KNN) Regression

**Data Preprocessing**:
- Feature scaling using `StandardScaler` (critical for KNN)
- Normalized all features to same scale

**Position Player KNN**:
- `KNeighborsRegressor` with n_neighbors=5
- Weights: 'uniform' (equal weight to all neighbors)
- Training/validation split: 80/20

**Pitcher KNN**:
- Same hyperparameters as position players
- Evaluated on pitcher-specific features

### 6. Model Comparison & Validation

**Final Evaluation**:
- Compared validation RMSE across all model types:
  - Multiple Linear Regression (All variables, Forward, Backward)
  - Decision Tree (Optimal depth)
  - K-Nearest Neighbors
- Separate comparisons for position players and pitchers
- Bar plots with labeled RMSE values

**Best Model Selection**:
- Identified lowest validation RMSE for each player category
- Considered model interpretability and complexity

## Technologies Used

- **Python 3**: Primary programming language
- **Jupyter Notebook**: Interactive analysis environment
- **Libraries**:
  - `pybaseball`: MLB data retrieval from FanGraphs and Baseball Reference
  - `pandas`: Data manipulation and analysis
  - `numpy`: Numerical computing
  - `seaborn`: Statistical data visualization
  - `matplotlib`: Plotting and visualization
  - `scikit-learn`: Machine learning models and evaluation
    - LinearRegression, DecisionTreeRegressor, KNeighborsRegressor
    - train_test_split, StandardScaler
    - mean_squared_error, r2_score
  - `mlxtend`: Sequential feature selection (forward/backward)
  - `sklearn.tree`: Decision tree visualization

## Key Findings

### Salary Patterns

- **Handedness Effect**: Batting handedness shows measurable impact on AAV
- **Seasonal Trends**: Contract values relatively stable 2022-2023, with variation in 2024
- **WAR Correlation**: Strong relationship between WAR (Wins Above Replacement) and AAV
- **Position vs. Pitchers**: Different salary distributions between player categories

### Model Performance

**Position Players**:
- Forward selection MLR showed improved performance over all-variable model
- Decision tree with depth=4 balanced complexity and accuracy
- KNN provided competitive predictions with proper scaling

**Pitchers**:
- Simpler decision tree (depth=2) optimal for pitchers
- Feature selection improved MLR validation performance
- Different optimal hyperparameters compared to position players

### Feature Importance

**Position Players - Key Predictors**:
- Service Time (experience affects market value)
- Advanced metrics: WAR, wRC+, wOBA
- Traditional stats: HR, RBI (still valued in free agency)
- Contact quality metrics: Hard%, LD%

**Pitchers - Key Predictors**:
- WAR and FIP (Fielding Independent Pitching)
- Strikeout metrics: K/9, K%
- Innings pitched and durability
- Advanced ERA metrics: xFIP

## Project Structure

```
.
├── Final_project_code.ipynb          # Main analysis notebook (210 cells)
├── Project_data_gathering.ipynb      # Data acquisition pipeline (142 cells)
└── README.md                          # Project documentation
```

## Statistical Techniques

- **Multiple Linear Regression**: Baseline predictive modeling
- **Sequential Feature Selection**: Forward and backward elimination
- **Cross-Validation**: 5-fold CV for feature selection
- **Decision Tree Regression**: Non-linear modeling with interpretability
- **K-Nearest Neighbors**: Instance-based regression
- **Hyperparameter Tuning**: Grid search for optimal tree depth
- **Feature Scaling**: StandardScaler for distance-based algorithms
- **Validation**: Hold-out validation (80/20 split)

## Evaluation Metrics

- **RMSE (Root Mean Squared Error)**: Primary metric for model comparison
- **R² (Coefficient of Determination)**: Explained variance
- **Training vs. Validation**: Assessed overfitting/underfitting
- **Feature Importance**: Variable contribution to predictions

## Insights for Baseball Operations

1. **Contract Negotiation**: Data-driven baselines for free agent offers
2. **Market Inefficiencies**: Identify over/undervalued players relative to predicted AAV
3. **Roster Construction**: Budget allocation based on predicted salaries
4. **Scouting Focus**: High-importance features guide player evaluation priorities
5. **Arbitration Cases**: Statistical support for salary arguments

## Use Cases

**For MLB Teams**:
- Set internal valuations before free agency negotiations
- Identify bargain free agents (actual AAV < predicted AAV)
- Budget planning for multi-year roster construction

**For Agents**:
- Establish market value baselines for client negotiations
- Demonstrate statistical comparables for salary justification

**For Analysts**:
- Study market efficiency in MLB free agency
- Evaluate which statistics teams value most

## Recommendations

1. **Feature Engineering**: Create interaction terms (e.g., WAR × Service Time)
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Position-Specific Models**: Separate models for catchers, infielders, outfielders
4. **Temporal Analysis**: Account for salary inflation year-over-year
5. **External Factors**: Integrate team payroll context and market size

## Future Enhancements

- Incorporate injury history and durability metrics
- Add team-specific effects (large vs. small market)
- Time series modeling of salary trends
- Integration of Statcast data (exit velocity, launch angle)
- Classification models for contract type (1-year vs. multi-year)
- Clustering analysis to identify player archetypes

## Limitations

1. **Sample Size**: Limited to 2022-2024 free agents (market may vary by year)
2. **Missing Contracts**: Players without contracts excluded from analysis
3. **Market Context**: Doesn't account for team needs, competition, or luxury tax
4. **Positional Differences**: Single model for all position players may mask position-specific patterns
5. **External Factors**: Player age, injury history not fully captured

## Academic Context

**Course**: Data Mining
**Institution**: George Washington University
**Date**: December 2024

## Contributors

- Matthew Wolf

## References

1. FanGraphs: Advanced baseball statistics and contract database
2. pybaseball: Python library for MLB data retrieval
3. mlxtend: Machine learning extensions for feature selection

## License

This project was completed as part of academic coursework at George Washington University. Baseball data sourced from FanGraphs and pybaseball under their respective terms of use.

## Data Attribution

Player statistics and contract data provided by FanGraphs and accessed via the pybaseball library, which aggregates publicly available MLB data from multiple sources including Baseball Reference and Baseball Savant.
