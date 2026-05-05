# Data-Science-Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Fraud Detection Machine Learning Project - A comprehensive data science project for detecting credit card fraud using advanced ML techniques.

## Project Organization
.
├── LICENSE <- Open-source license if one is chosen (MIT, Apache 2.0, etc.)
├── Makefile <- Makefile with convenience commands like make data, make train, make test
├── README.md <- The top-level README for developers using this project
├── requirements.txt <- The requirements file for reproducing the analysis environment
│ generated with pip freeze > requirements.txt
├── pyproject.toml <- Project configuration file with package metadata and build system
├── poetry.lock <- Poetry lock file for reproducible dependency resolution
├── .env.example <- Example environment variables file (copy to .env and configure)
├── .gitignore <- Git ignore file specifying intentionally untracked files to ignore
│
├── .github <- GitHub-specific files (workflows, issue templates, etc.)
│ └── workflows <- GitHub Actions CI/CD pipelines
│
├── api <- REST API for model serving and predictions
│ ├── DockerFile <- Docker configuration for containerizing the API service
│ ├── init.py <- Makes api a Python package
│ ├── config.py <- API configuration settings (host, port, environment variables)
│ ├── engine.py <- Core API engine and request handling logic
│ ├── extract_constants.py <- Constants and configuration extraction utilities
│ ├── main.py <- FastAPI/Flask application entry point and route definitions
│ └── schemas.py <- Pydantic models for request/response validation
│
├── dashboard <- Interactive dashboard application (Streamlit/Dash/Plotly)
│ for visualizing model performance and data insights
│
├── data <- Data directory following Cookiecutter Data Science structure
│ ├── pycache <- Python bytecode cache (auto-generated, should be gitignored)
│ ├── filters.py <- Data filtering and query functions
│ ├── loaders.py <- Data loading utilities for different formats (CSV, Parquet, etc.)
│ │
│ ├── external <- Data from third party sources (immutable)
│ │ └── .gitkeep <- Placeholder to keep directory in git
│ │
│ ├── interim <- Intermediate data that has been transformed
│ │ ├── cleaned <- Cleaned datasets after data validation and cleaning
│ │ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ │ └── credit_card_fraud_10k_cleaned.csv <- Cleaned fraud detection dataset
│ │ │
│ │ ├── label <- Label files (target variables) for train/val/test splits
│ │ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ │ ├── Y_test.csv <- Test set labels (target variable)
│ │ │ ├── Y_train.csv <- Training set labels (target variable)
│ │ │ └── Y_val.csv <- Validation set labels (target variable)
│ │ │
│ │ ├── scaled <- Scaled/normalized features for modeling
│ │ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ │ ├── X_test.csv <- Scaled test features
│ │ │ ├── X_train.csv <- Scaled training features
│ │ │ └── X_val.csv <- Scaled validation features
│ │ │
│ │ ├── smote <- Data processed with SMOTE for handling class imbalance
│ │ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ │ └── X_train_smote_eda.csv <- SMOTE-augmented training data for EDA
│ │ │
│ │ └── unscaled <- Original unscaled features (post-preprocessing)
│ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ ├── X_test.csv <- Unscaled test features
│ │ ├── X_train.csv <- Unscaled training features
│ │ └── X_val.csv <- Unscaled validation features
│ │
│ ├── processed <- The final, canonical data sets for modeling
│ │ └── .gitkeep <- Placeholder to keep directory in git
│ │
│ ├── raw <- The original, immutable data dump (never edit these files)
│ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ └── credit_card_fraud_10k.csv <- Original raw credit card fraud dataset
│ │
│ └── .gitkeep <- Placeholder to keep data directory in git
│
├── docs <- Documentation directory using MkDocs
│ ├── docs <- Documentation source files (Markdown)
│ ├── README.md <- Documentation README
│ └── mkdocs.yml <- MkDocs configuration file
│
├── models <- Trained and serialized models, model predictions, or model summaries
│ ├── .gitkeep <- Placeholder to keep directory in git
│ └── final_fraud_detection_model.pkl <- Final production-ready fraud detection model (pickle)
│
├── notebooks <- Jupyter notebooks for exploratory analysis and prototyping
│ │ Naming convention: (number)_(description).ipynb
│ │
│ ├── mlruns <- MLflow tracking directory for experiment management
│ │ └── .gitkeep <- Placeholder to keep directory in git
│ │
│ ├── 01_data_validation_and_cleaning.ipynb <- Data validation, quality checks, and cleaning
│ ├── 02_exploratory_data_analysis_eda.ipynb <- Exploratory data analysis and visualization
│ ├── 03_preprocessing_and_feature_engineering.ipynb <- Feature preprocessing and engineering
│ ├── 04_eda_post_feature_engineering.ipynb <- EDA after feature engineering
│ ├── 05_model_training_mlflow.ipynb <- Model training with MLflow experiment tracking
│ └── 06_model_evaluation_selection.ipynb <- Model evaluation, comparison, and selection
│
├── outputs <- Generated analysis outputs and artifacts from the project
│ └── model_selection <- Model selection and evaluation outputs
│ ├── csv <- CSV files with model metrics and comparisons
│ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ ├── all_models_comparison_final.csv <- Comparison metrics for all trained models
│ │ ├── final_model_evaluation_results.csv <- Final evaluation results for selected model
│ │ └── final_model_recommendation.csv <- Model recommendation with justification
│ │
│ ├── images <- Visualizations and plots for model evaluation
│ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ ├── confusion_matrices.png <- Confusion matrices for model performance
│ │ └── final_model_evaluation_charts.png <- Evaluation charts (ROC, PR curves, etc.)
│ │
│ ├── json <- JSON files with structured output data
│ │ ├── .gitkeep <- Placeholder to keep directory in git
│ │ └── model_deployment_info.json <- Model metadata and deployment configuration
│ │
│ └── text <- Text-based reports and summaries
│ ├── .gitkeep <- Placeholder to keep directory in git
│ └── final_top_3_models.txt <- Summary of top 3 performing models
│
├── pages <- Dashboard or web application pages (Streamlit/Dash pages)
│ ├── bivariate.py <- Bivariate analysis page for two-variable relationships
│ ├── custom.py <- Custom analysis page for user-defined visualizations
│ ├── multivariate.py <- Multivariate analysis page for complex relationships
│ ├── temporal.py <- Temporal/time-series analysis page
│ └── univariate.py <- Univariate analysis page for single-variable distributions
│
├── references <- Data dictionaries, manuals, and all other explanatory materials
│ └── .gitkeep <- Placeholder to keep directory in git
│
├── reports <- Generated analysis reports (HTML, PDF, LaTeX, etc.)
│ ├── figures <- Generated graphics and figures for reports
│ │ └── .gitkeep <- Placeholder to keep directory in git
│ │
│ ├── .gitkeep <- Placeholder to keep directory in git
│ └── validation_report.json <- Data validation report with quality metrics
│
├── scaler <- Serialized scalers, transformers, and preprocessors
│ ├── .gitkeep <- Placeholder to keep directory in git
│ └── scaler.pkl <- Fitted scaler/transformer for feature preprocessing (pickle)
│
├── tests <- Unit and integration tests for the project
│ ├── test_data.py <- Tests for data loading, cleaning, and validation functions
│ └── test_pipeline.py <- Tests for ML pipeline components and end-to-end workflows
│
└── src <- Source code for use in this project (main Python package)
├── init.py <- Makes src a Python package
├── pycache <- Python bytecode cache (auto-generated, should be gitignored)
├── app.py <- Main application entry point (dashboard or API)
├── config.py <- Global configuration and constants for the project
│
├── data <- Data processing and management module
│ ├── init.py <- Makes data a Python package
│ ├── pycache <- Python bytecode cache
│ ├── clean.py <- Data cleaning functions (handle missing values, outliers, etc.)
│ ├── inspect_data.py <- Data inspection and profiling utilities
│ ├── load_data.py <- Data loading functions for various sources and formats
│ └── validate.py <- Data validation functions (schema validation, quality checks)
│
├── features <- Feature engineering module
│ ├── init.py <- Makes features a Python package
│ ├── encoding.py <- Categorical encoding functions (one-hot, label, target encoding)
│ ├── engineering.py <- Feature engineering logic (create new features, transformations)
│ ├── save.py <- Feature saving utilities (save processed features to disk)
│ ├── scaling.py <- Feature scaling functions (standardization, normalization, etc.)
│ ├── split.py <- Train/validation/test split utilities
│ ├── utils.py <- Feature engineering utility functions
│ └── validation.py <- Feature validation (check for leakage, quality, etc.)
│
├── models <- Machine learning models module
│ ├── init.py <- Makes models a Python package
│ ├── config.py <- Model configuration and hyperparameters
│ ├── evaluation.py <- Model evaluation functions (metrics, cross-validation, etc.)
│ ├── loader.py <- Model loading utilities for deployment
│ ├── train.py <- Model training logic and pipelines
│ ├── tune.py <- Hyperparameter tuning utilities (GridSearch, RandomSearch, etc.)
│ └── utils.py <- Model utility functions (save, load, serialize, etc.)
│
├── utils <- General utility functions
│ ├── init.py <- Makes utils a Python package
│ ├── pycache <- Python bytecode cache
│ └── plotting.py <- Plotting and visualization utility functions
│
└── visualizations <- Visualization and plotting module
├── init.py <- Makes visualizations a Python package
├── pycache <- Python bytecode cache
├── bivariate_categorical.py <- Bivariate plots for categorical variables
├── bivariate_continuous.py <- Bivariate plots for continuous variables
├── bivariate_one.py <- General bivariate analysis functions
├── data_loader.py <- Data loading for visualizations
├── feature_interactions_advanced.py <- Advanced feature interaction plots
├── interaction.py <- Feature interaction visualizations
├── multivariate.py <- Multivariate visualization functions
├── post_feature_engineer_data_loader.py <- Data loader for post-engineering viz
├── risk_analysis.py <- Risk analysis visualizations
├── temporal.py <- Time-series and temporal visualizations
└── univariate.py <- Univariate distribution plots


## Quick Start

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` or `poetry install`
3. Copy `.env.example` to `.env` and configure your environment variables
4. Explore the notebooks in order (01-06)
5. Run tests: `pytest tests/` or `make test`
6. Run the dashboard: `streamlit run app.py` (or appropriate command)
7. Deploy the API: `docker build -t fraud-api . && docker run -p 8000:8000 fraud-api`

## Data Pipeline

The data flows through the following stages:
- **raw**: Original immutable data
- **interim/cleaned**: Validated and cleaned data
- **interim/unscaled**: Preprocessed but unscaled features
- **interim/scaled**: Scaled/normalized features ready for modeling
- **interim/smote**: Data augmented with SMOTE for class imbalance
- **processed**: Final canonical datasets for production

## Model Development

1. Exploratory Data Analysis (notebooks 01-04)
2. Feature Engineering (src/features/)
3. Model Training with MLflow (notebook 05)
4. Model Evaluation and Selection (notebook 06)
5. Final model saved to `models/` directory
6. Scaler saved to `scaler/` directory

## Testing

Run the test suite using:
```bash
pytest tests/
# or
make test
```
--------

<p><small>Project generated using <a href="https://cookiecutter-data-science.drivendata.org/" target="_blank">Cookiecutter Data Science</a> template.</small></p>
