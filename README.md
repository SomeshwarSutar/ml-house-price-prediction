# House Price Prediction with Linear Regression

An interactive Streamlit application that demonstrates multiple linear
regression with the Kaggle Ames Housing dataset. The app trains a scikit-learn
model when it starts, evaluates it on held-out data, and lets users estimate a
sale price from six house characteristics.

This project is intended for learning and demonstration. Its predictions are
based on historical home sales in Ames, Iowa, and are not current property
valuations or financial advice.

## Features

- Enter house characteristics and receive an estimated sale price in US dollars.
- Inspect held-out validation metrics: R-squared, MAE, and RMSE.
- Compare actual and predicted prices in an interactive Plotly chart.
- Review standardized model coefficients and their directions.
- Explore relationships between each selected feature and `SalePrice`.
- Cache data loading and model training with Streamlit to avoid unnecessary work.

## Dataset

The project uses `train.csv` from Kaggle's
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
competition. The dataset describes residential property sales in Ames, Iowa.

The model intentionally uses a small, understandable subset of the available
columns:

| Column | Meaning | Role |
| --- | --- | --- |
| `OverallQual` | Overall material and finish quality, from 1 to 10 | Feature |
| `GrLivArea` | Above-ground living area in square feet | Feature |
| `YearBuilt` | Original construction year | Feature |
| `FullBath` | Number of full bathrooms above ground | Feature |
| `BedroomAbvGr` | Number of bedrooms above ground | Feature |
| `GarageCars` | Garage capacity in cars | Feature |
| `SalePrice` | Property sale price in US dollars | Target |

The full competition data dictionary is available in
[`data_description.txt`](data_description.txt).

## How the Model Works

### Multiple linear regression

Linear regression estimates a continuous target as a weighted sum of input
features. Because this project uses six input features, it is a **multiple
linear regression** model:

$$
\hat{y} = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_6x_6
$$

Here, $\beta_0$ is the intercept, each $\beta_i$ is a learned coefficient, and
each $x_i$ is a house feature. During fitting, scikit-learn's
`LinearRegression` chooses coefficients that minimize the residual sum of
squares:

$$
\min_{\beta}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

A positive coefficient is associated with a higher prediction when the other
features remain fixed; a negative coefficient is associated with a lower
prediction. This is an association learned from the data, not evidence that a
feature causes a price change.

### The model used by this project

House prices are right-skewed, so this app does not fit the regression directly
to raw dollar values. It transforms the target with `numpy.log1p`:

$$
z = \log(1 + SalePrice)
$$

The fitted equation is therefore:

$$
\widehat{z} = \beta_0 + \beta_1x_1 + \cdots + \beta_6x_6
$$

After prediction, `numpy.expm1` transforms the result back to dollars:

$$
\widehat{SalePrice} = \exp(\widehat{z}) - 1
$$

This target transformation reduces the influence of very expensive outliers
and makes relative price differences easier for a linear model to represent.
Because the inverse transformation is nonlinear, the final dollar prediction
is not a simple linear sum of the original feature values.

### Training pipeline

The implementation uses these scikit-learn steps:

1. `train_test_split` reserves 20% of the rows for validation and uses
	 `random_state=42` for a repeatable split.
2. `SimpleImputer(strategy="median")` replaces any missing feature values with
	 medians learned from the training data.
3. `StandardScaler` centers each feature and scales it to unit variance.
4. `LinearRegression` learns the intercept and coefficients by ordinary least
	 squares.
5. `TransformedTargetRegressor` applies `log1p` to training targets and
	 automatically applies `expm1` to predictions.
6. Negative predictions, which are not meaningful for sale prices, are clipped
	 to zero for display and validation.

The imputer, scaler, and regressor are combined in a `Pipeline`. This is
important because preprocessing parameters are learned only from the training
partition, preventing validation data from leaking into model fitting.

Standardization does not generally change an ordinary least-squares model's
predictions, but it puts the coefficients on comparable input scales. The
coefficient chart can therefore show relative direction and influence more
clearly. Coefficients still should not be interpreted as causal effects or as
direct dollar changes because the target is logarithmic.

### Evaluation metrics

The app evaluates predictions against the held-out 20% validation partition:

- **R-squared ($R^2$)** measures the proportion of target variance explained by
	the model. A score near 1 is better; 0 means the model is no better than
	predicting the validation mean, and a score can be negative.
- **Mean absolute error (MAE)** is the average absolute difference between the
	actual and predicted prices. It is displayed in dollars and is relatively
	easy to interpret.
- **Root mean squared error (RMSE)** is the square root of the average squared
	error. It is also displayed in dollars, but penalizes large errors more than
	MAE does.

The actual-versus-predicted chart includes a diagonal reference line. Points
close to that line are more accurate; points far from it have larger residuals.

## Application Views

- **Make a prediction** collects the six model inputs in a form and displays an
	estimated sale price.
- **Model performance** shows validation metrics and the actual-versus-predicted
	scatter plot.
- **How it works** presents the regression equation and standardized
	coefficients.
- **Explore data** displays sample rows, descriptive statistics, and feature
	relationships with ordinary least-squares trend lines.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for the commands shown below
- Kaggle credentials and acceptance of the competition rules only when the
	dataset needs to be downloaded

The main libraries are Streamlit, pandas, NumPy, scikit-learn, Plotly, and
statsmodels. Plotly uses statsmodels to calculate the OLS trend lines in the
data exploration view.

## Installation and Usage

1. Clone the repository and enter its directory.

2. Install the locked project dependencies:

	 ```powershell
	 uv sync
	 ```

3. Confirm that `data/train.csv` exists. If it does not, authenticate the
	 Kaggle CLI, accept the competition rules on Kaggle, and run:

	 ```powershell
	 uv run python download_data.py
	 ```

	 Kaggle supports credentials through its configuration file or environment
	 variables. Do not commit API credentials to this repository.

4. Start the application:

	 ```powershell
	 uv run streamlit run app.py
	 ```

5. Open the local URL printed by Streamlit, normally
	 `http://localhost:8501`.

The model is fitted from `data/train.csv` in memory when the application first
runs. It does not create or load a serialized model file.

## Project Structure

```text
.
|-- app.py                  # Streamlit UI, preprocessing, training, and evaluation
|-- download_data.py        # Kaggle download and extraction helper
|-- pyproject.toml          # Project metadata and Python dependencies
|-- data_description.txt    # Ames Housing column descriptions
|-- data/
|   `-- train.csv           # Training data required by the app
`-- README.md
```

The repository may also contain Kaggle's `test.csv` and
`sample_submission.csv`, but the application does not use them. The download
helper removes those two files after extraction to keep only the data needed by
this demonstration.

## Implementation Map

- `load_data()` reads the CSV, verifies that all required columns exist, and
	returns only the selected features and target.
- `train_model()` creates the split and pipeline, fits the transformed-target
	model, calculates validation metrics, and prepares chart data.
- `calculate_coefficients()` extracts fitted standardized coefficients for the
	explanatory chart.
- `currency_usd()` formats predictions and errors for display.
- `@st.cache_data` caches CSV loading, while `@st.cache_resource` caches the
	trained model bundle.

## Limitations

- The six-feature model omits many relevant factors, including neighborhood,
	lot size, condition, renovation, and market timing.
- The dataset represents historical sales in Ames, Iowa, so the model should
	not be generalized to other locations or present-day markets.
- Linear regression assumes that predictors combine linearly on the
	log-price scale. Real housing relationships can include nonlinear effects and
	interactions.
- A single train/validation split gives a less stable performance estimate than
	cross-validation.
- Correlated inputs can make individual coefficients unstable even when overall
	predictions remain useful.
- The model provides point estimates only; it does not calculate prediction
	intervals or uncertainty.

Possible next steps include cross-validation, additional feature engineering,
categorical preprocessing with `ColumnTransformer`, regularized linear models
such as Ridge or Lasso, and comparison with tree-based regressors.
