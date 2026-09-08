from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="House Price Linear Regression",
    layout="wide",
)

DATA_PATH = Path(__file__).resolve().parent / "data" / "train.csv"

FEATURES = [
    "OverallQual",
    "GrLivArea",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "GarageCars",
]

TARGET = "SalePrice"

FEATURE_LABELS = {
    "OverallQual": "Overall quality",
    "GrLivArea": "Above-ground living area",
    "YearBuilt": "Year built",
    "FullBath": "Full bathrooms",
    "BedroomAbvGr": "Bedrooms above ground",
    "GarageCars": "Garage capacity",
}

FEATURE_UNITS = {
    "OverallQual": "quality points",
    "GrLivArea": "square feet",
    "YearBuilt": "years",
    "FullBath": "bathrooms",
    "BedroomAbvGr": "bedrooms",
    "GarageCars": "cars",
}


# ---------------------------------------------------------
# Data and model functions
# ---------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and validate the Kaggle training data."""
    data = pd.read_csv(path)

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The dataset is missing these columns: "
            + ", ".join(missing_columns)
        )

    return data[required_columns].copy()


@st.cache_resource
def train_model(data: pd.DataFrame) -> dict:
    """
    Train a Linear Regression pipeline.

    The feature pipeline:
    1. Replaces missing numerical values with the median.
    2. Standardizes the features.
    3. Trains Linear Regression.

    The target is log-transformed during training and converted
    back to dollars automatically during prediction.
    """
    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    feature_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )

    model = TransformedTargetRegressor(
        regressor=feature_pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    validation_predictions = np.maximum(validation_predictions, 0)

    metrics = {
        "r2": r2_score(y_validation, validation_predictions),
        "mae": mean_absolute_error(
            y_validation,
            validation_predictions,
        ),
        "rmse": np.sqrt(
            mean_squared_error(
                y_validation,
                validation_predictions,
            )
        ),
    }

    evaluation = pd.DataFrame(
        {
            "Actual price": y_validation.to_numpy(),
            "Predicted price": validation_predictions,
        }
    )

    return {
        "model": model,
        "metrics": metrics,
        "evaluation": evaluation,
        "training_rows": len(X_train),
        "validation_rows": len(X_validation),
    }


def currency_usd(value: float) -> str:
    """Display a value as US dollars."""
    return f"${value:,.0f}"


def calculate_coefficients(model_bundle: dict) -> pd.DataFrame:
    """
    Return standardized model coefficients.

    Because the target is log-transformed and inputs are standardized,
    these coefficients are primarily useful for comparing direction
    and relative influence, not direct dollar interpretation.
    """
    target_regressor = model_bundle["model"]
    fitted_pipeline = target_regressor.regressor_

    coefficients = fitted_pipeline.named_steps[
        "regressor"
    ].coef_

    coefficient_data = pd.DataFrame(
        {
            "Feature": [
                FEATURE_LABELS[feature] for feature in FEATURES
            ],
            "Coefficient": coefficients,
        }
    )

    coefficient_data["Direction"] = np.where(
        coefficient_data["Coefficient"] >= 0,
        "Raises prediction",
        "Lowers prediction",
    )

    coefficient_data["Absolute influence"] = (
        coefficient_data["Coefficient"].abs()
    )

    return coefficient_data.sort_values(
        "Absolute influence",
        ascending=True,
    )


# ---------------------------------------------------------
# Load project assets
# ---------------------------------------------------------

st.title("House Price Prediction with Linear Regression")

st.caption(
    "An educational machine-learning demonstration using "
    "the Kaggle Ames Housing dataset."
)

if not DATA_PATH.exists():
    st.error(
        "The Kaggle dataset was not found at "
        f"`{DATA_PATH}`."
    )
    st.code("python download_data.py", language="bash")
    st.info(
        "Authenticate with Kaggle and accept the competition "
        "rules before running the download script."
    )
    st.stop()

try:
    housing_data = load_data(str(DATA_PATH))
    model_bundle = train_model(housing_data)
except Exception as error:
    st.error(f"Unable to prepare the application: {error}")
    st.stop()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("About the model")

    st.write(
        "This is a multiple Linear Regression model because "
        "it uses several input variables to predict one "
        "continuous value."
    )

    st.metric("Dataset rows", f"{len(housing_data):,}")
    st.metric("Training rows", f"{model_bundle['training_rows']:,}")
    st.metric(
        "Validation rows",
        f"{model_bundle['validation_rows']:,}",
    )

    st.divider()

    st.warning(
        "The data represents Ames, Iowa. Predictions are in "
        "US dollars and must not be treated as current Indian "
        "property valuations or professional financial advice."
    )


# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------

prediction_tab, performance_tab, explanation_tab, data_tab = st.tabs(
    [
        "Make a prediction",
        "Model performance",
        "How it works",
        "Explore data",
    ]
)


# ---------------------------------------------------------
# Prediction tab
# ---------------------------------------------------------

with prediction_tab:
    st.subheader("Enter the house characteristics")

    medians = housing_data[FEATURES].median()

    with st.form("house_prediction_form"):
        first_column, second_column = st.columns(2)

        with first_column:
            overall_quality = st.slider(
                "Overall material and finish quality",
                min_value=1,
                max_value=10,
                value=int(medians["OverallQual"]),
                help="1 means very poor and 10 means excellent.",
            )

            living_area = st.number_input(
                "Above-ground living area, square feet",
                min_value=300,
                max_value=6000,
                value=int(medians["GrLivArea"]),
                step=50,
            )

            year_built = st.number_input(
                "Year built",
                min_value=1870,
                max_value=2026,
                value=int(medians["YearBuilt"]),
                step=1,
            )

        with second_column:
            full_bathrooms = st.number_input(
                "Full bathrooms",
                min_value=0,
                max_value=5,
                value=int(medians["FullBath"]),
                step=1,
            )

            bedrooms = st.number_input(
                "Bedrooms above ground",
                min_value=0,
                max_value=10,
                value=int(medians["BedroomAbvGr"]),
                step=1,
            )

            garage_capacity = st.number_input(
                "Garage capacity, number of cars",
                min_value=0,
                max_value=5,
                value=int(medians["GarageCars"]),
                step=1,
            )

        submitted = st.form_submit_button(
            "Predict sale price",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_data = pd.DataFrame(
            [
                {
                    "OverallQual": overall_quality,
                    "GrLivArea": living_area,
                    "YearBuilt": year_built,
                    "FullBath": full_bathrooms,
                    "BedroomAbvGr": bedrooms,
                    "GarageCars": garage_capacity,
                }
            ]
        )

        predicted_price = model_bundle["model"].predict(
            input_data
        )[0]

        predicted_price = max(float(predicted_price), 0.0)

        st.success("Prediction completed")
        st.metric(
            "Estimated sale price",
            currency_usd(predicted_price),
        )

        st.caption(
            "This estimate is based on historical Ames Housing "
            "records and is intended only to demonstrate "
            "Linear Regression."
        )

        with st.expander("View model input"):
            display_input = input_data.rename(
                columns=FEATURE_LABELS
            )
            st.dataframe(
                display_input,
                hide_index=True,
                use_container_width=True,
            )


# ---------------------------------------------------------
# Performance tab
# ---------------------------------------------------------

with performance_tab:
    st.subheader("Held-out validation results")

    metrics = model_bundle["metrics"]

    metric_column_1, metric_column_2, metric_column_3 = st.columns(3)

    metric_column_1.metric(
        "R² score",
        f"{metrics['r2']:.3f}",
        help=(
            "Proportion of variation in sale price explained "
            "by the selected input features."
        ),
    )

    metric_column_2.metric(
        "Mean absolute error",
        currency_usd(metrics["mae"]),
        help="Average absolute difference from the actual price.",
    )

    metric_column_3.metric(
        "Root mean squared error",
        currency_usd(metrics["rmse"]),
        help="An error measure that penalizes larger mistakes.",
    )

    evaluation = model_bundle["evaluation"]

    minimum_value = min(
        evaluation["Actual price"].min(),
        evaluation["Predicted price"].min(),
    )

    maximum_value = max(
        evaluation["Actual price"].max(),
        evaluation["Predicted price"].max(),
    )

    actual_vs_predicted = px.scatter(
        evaluation,
        x="Actual price",
        y="Predicted price",
        title="Actual versus predicted house prices",
        opacity=0.65,
    )

    actual_vs_predicted.add_shape(
        type="line",
        x0=minimum_value,
        y0=minimum_value,
        x1=maximum_value,
        y1=maximum_value,
        line={
            "color": "red",
            "dash": "dash",
        },
    )

    actual_vs_predicted.update_layout(
        xaxis_tickprefix="$",
        yaxis_tickprefix="$",
    )

    st.plotly_chart(
        actual_vs_predicted,
        use_container_width=True,
    )

    st.info(
        "Points close to the red diagonal line represent "
        "accurate predictions. Large distances from the line "
        "represent larger prediction errors."
    )


# ---------------------------------------------------------
# Explanation tab
# ---------------------------------------------------------

with explanation_tab:
    st.subheader("Linear Regression mechanics")

    st.latex(
        r"""
        \log(1 + \widehat{SalePrice}) =
        \beta_0 +
        \beta_1 x_1 +
        \beta_2 x_2 +
        \cdots +
        \beta_6 x_6
        """
    )

    st.write(
        """
        The model learns an intercept and one coefficient for
        each input feature. A positive coefficient moves the
        prediction upward, while a negative coefficient moves
        it downward, assuming the other features stay fixed.
        """
    )

    coefficients = calculate_coefficients(model_bundle)

    coefficient_chart = px.bar(
        coefficients,
        x="Coefficient",
        y="Feature",
        color="Direction",
        orientation="h",
        title="Standardized feature coefficients",
        color_discrete_map={
            "Raises prediction": "#2E8B57",
            "Lowers prediction": "#C0392B",
        },
    )

    st.plotly_chart(
        coefficient_chart,
        use_container_width=True,
    )

    st.warning(
        "Coefficient direction shows association in this model, "
        "not causation. Inputs such as quality, size, and year "
        "built can also be correlated with one another."
    )

    st.markdown(
        """
        **Model pipeline**

        1. Missing numerical values are replaced with medians.
        2. Input features are standardized.
        3. `SalePrice` is transformed with `log1p`.
        4. Linear Regression learns the coefficients.
        5. Predictions are transformed back into US dollars.
        """
    )


# ---------------------------------------------------------
# Data exploration tab
# ---------------------------------------------------------

with data_tab:
    st.subheader("Explore the Kaggle training data")

    st.dataframe(
        housing_data.head(20),
        use_container_width=True,
    )

    selected_feature = st.selectbox(
        "Select a feature to compare with sale price",
        options=FEATURES,
        format_func=lambda value: FEATURE_LABELS[value],
    )

    relationship_chart = px.scatter(
        housing_data,
        x=selected_feature,
        y=TARGET,
        trendline="ols",
        title=(
            f"{FEATURE_LABELS[selected_feature]} "
            "versus sale price"
        ),
        opacity=0.55,
    )

    relationship_chart.update_layout(
        yaxis_tickprefix="$",
    )

    st.plotly_chart(
        relationship_chart,
        use_container_width=True,
    )

    with st.expander("View descriptive statistics"):
        st.dataframe(
            housing_data.describe().T,
            use_container_width=True,
        )