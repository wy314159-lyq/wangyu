# AutoMatFlow: Automated Feature Selection and Modeling Workflow

AutoMatFlow is a Python script designed to automate and streamline the process of feature selection and model evaluation for tabular data, particularly suited for materials science or similar domains but applicable generally. It offers a range of techniques for data preprocessing, feature filtering, advanced feature selection, hyperparameter optimization, model training, and interpretation using SHAP.

## Key Features

*   **Data Handling:** Reads data from Excel (.xlsx, .xls) or CSV (.csv) files.By default, the last column is treated as the target variable. Additionally, to minimize potential errors, it is recommended to use English characters and avoid special symbols.
*   **Task Types:** Supports both Regression and Classification tasks.
*   **Preprocessing:**
    *   Handles missing values (imputation using mean or most frequent).
    *   Feature scaling (StandardScaler).
    *   One-Hot Encoding for specified numerical categorical features.
    *   Optional Gaussian Noise addition to numerical features for robustness.
*   **Model Support:** Includes a wide variety of scikit-learn models:
    *   Linear Models (Linear/Logistic Regression, Ridge, Lasso, ElasticNet)
    *   Tree-based Models (Decision Tree, Random Forest, Gradient Boosting, AdaBoost, ExtraTrees, HistGradientBoosting)
    *   Support Vector Machines (SVR, SVC)
    *   K-Nearest Neighbors (KNN)
    *   Neural Networks (MLP)
    *   Gaussian Processes (GP)
    *   **Optional:** XGBoost, LightGBM, CatBoost, TabPFN (requires separate installation).
*   **Hyperparameter Optimization:** Uses Optuna for efficient Bayesian optimization to find the best model parameters.
*   **Feature Selection Pipeline:**
    *   **Initial Filtering:** Based on model feature importance and high correlation removal.
    *   **Advanced Selection:** Choose between Exhaustive Search (evaluating combinations with the optimized pipeline), Recursive Feature Elimination (RFE), or Genetic Algorithm (GA) to find the optimal feature subset.
*   **Evaluation:**
    *   Robust cross-validation (KFold, StratifiedKFold, GroupKFold options inferred).
    *   Standard performance metrics (RMSE, R², Accuracy, F1, Precision, Recall, AUC).
    *   Evaluation on a held-out test set.
*   **Interpretation:** Integrated SHAP analysis for explaining the final model's predictions (summary plots, dependence plots, waterfall plots, etc.).
*   **GPU Acceleration:** Supports GPU usage for XGBoost, LightGBM, CatBoost, and TabPFN if compatible hardware and drivers are present and configured (`use_gpu` flag).
*   **Configuration:** Highly configurable via a central dictionary at the top of the script.
*   **Output:** Generates a timestamped results directory containing:
    *   Detailed logs and intermediate results (CSV files).
    *   Numerous plots (importance, correlation, optimization history, combination performance, final evaluation, SHAP plots).
    *   The final trained model pipeline saved as a `.pkl` file (`final_model_with_metadata.pkl`).

## Installation

1.  **Clone or Download:** Get the `AutoMatFlow.py` script.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:** Install the required packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note:** Some libraries like `lightgbm`, `catboost`, `xgboost`, and `tabpfn` provide significant modeling capabilities but might have specific installation requirements (e.g., build tools, CUDA for GPU versions). Install them individually if needed:
        ```bash
        pip install lightgbm xgboost catboost tabpfn
        # Or for GPU versions (check their respective documentation):
        # pip install xgboost --upgrade # Often includes GPU support if CUDA is set up
        # pip install lightgbm --install-option=--gpu # May require specific build steps
        # pip install catboost # GPU support often included
        # pip install tabpfn # Will use torch, which needs CUDA setup for GPU
        ```

## Configuration

**Crucially, before running the script, you MUST configure the settings at the top of `AutoMatFlow.py`.**

Key sections to configure:

*   **`DATA_CONFIG`**:
    *   `data_file`: Path to your dataset (`.xlsx`, `.xls`, `.csv`).
    *   `target_column`: Index of the target variable (use -1 for the last column).
    *   `categorical_num_cols`: **Important!** List of column *names* (strings) that contain numerical values but should be treated as categories (e.g., `["Material_Code", "Process_ID"]`). These will be one-hot encoded. If empty, all non-target columns are treated as continuous numerical features.
    *   Other parameters for file reading (sheet name, header row, separator, encoding).
*   **`BASIC_CONFIG`**:
    *   `task_type`: Set to `"regression"` or `"classification"`.
    *   `test_size`: Proportion of data for the test set (e.g., 0.2 for 20%).
    *   `random_state`: For reproducibility.
*   **`MODEL_CONFIG`**:
    *   `cv`: Number of folds for cross-validation during optimization and evaluation.
    *   `bayes_iter`: Number of iterations for Optuna hyperparameter search.
*   **`FEATURE_CONFIG`**:
    *   `importance_threshold`: Percentile threshold for initial importance filtering (e.g., 0.2 keeps top 80%).
    *   `correlation_threshold`: Maximum allowed pairwise correlation between features.
*   **`SHAP_CONFIG`**: Settings for SHAP analysis (sample size, plot types).
*   **`GPU_CONFIG`**:
    *   `use_gpu`: Set to `True` to *attempt* using GPU for supported models (XGBoost, LightGBM, CatBoost, TabPFN). Requires correct drivers and library installations. Defaults to `False`.
*   **`NOISE_CONFIG`**:
    *   `add_gaussian_noise`: `True` or `False` to add noise during preprocessing.
    *   `gaussian_noise_scale`: Standard deviation of noise relative to feature standard deviation.
*   **`Genetic Algorithm Config`**: Parameters for the GA feature selection method (population size, generations, rates).

## Usage

1.  **Configure:** Modify the configuration dictionaries at the top of `AutoMatFlow.py` to match your dataset and desired workflow.
2.  **Run:** Execute the script from your terminal:
    ```bash
    python AutoMatFlow.py
    ```
3.  **Interact:** The script will prompt you to:
    *   Confirm the task type (Regression/Classification).
    *   Choose a base model for optimization and evaluation from a list.
    *   Select an advanced feature selection method (Exhaustive, RFE, GA, or Skip).
    *   Decide whether to re-optimize hyperparameters for the final feature set.
    *   Confirm if you want to perform the final model evaluation.
    *   Confirm if you want to perform SHAP analysis.
4.  **Results:** Wait for the analysis to complete. A new directory named `feature_selection_results_YYYYMMDD_HHMMSS` will be created containing all outputs.

## Output Files

The results directory will contain various files, including:

*   `hyperparameter_optimization_history.csv/png`: Optuna optimization progress.
*   `feature_importance_results.csv`: Feature importance scores from the initial model.
*   `feature_importance.png`: Plot of feature importances.
*   `correlation_heatmap_before/after.png`: Feature correlation heatmaps.
*   `correlation_filtered_features.csv`: Features remaining after correlation filtering.
*   `feature_combinations_results.csv` / `rfe_results.csv` / `genetic_algorithm_results.csv`: Results from the chosen advanced feature selection method.
*   `top_feature_combinations.csv`: Top results from the feature selection.
*   `final_model_performance.csv`: Key performance metrics of the final model.
*   `final_model_evaluation.png`: Plot comparing actual vs. predicted values for the final model.
*   `confusion_matrix.png` / `roc_curve.png`: (Classification only) Evaluation plots.
*   `final_model_with_metadata.pkl`: The trained scikit-learn Pipeline object (including preprocessor and model) and selected feature names. You can load this for future predictions.
*   `shap_analysis/`: Subdirectory containing SHAP plots (`summary_plot.png`, `bar_plot.png`, `dependence_plot_*.png`, etc.) and importance values (`shap_feature_importance.csv`) if SHAP analysis was performed.


## Tips for Handling Overfitting

Overfitting occurs when a model learns the training data too well, including its noise and specific patterns, leading to excellent performance on the training set but poor performance on unseen data (like the test set). You can often detect overfitting by comparing the performance metrics (e.g., RMSE, R², F1, Accuracy) between the training set and the test set in the final evaluation output (`final_model_performance.csv`). A large gap indicates potential overfitting.

Here's how AutoMatFlow can help, and how you can adjust its configuration with **specific examples**:

1.  **Leverage Existing Features:**
    *   **Cross-Validation (`cv`):** The script uses CV during hyperparameter optimization and evaluation. This inherently helps select models and parameters that generalize better by evaluating performance on multiple internal validation sets. Ensure `cv` is set to a reasonable value (e.g., `5` or `10` in `MODEL_CONFIG`).
    *   **Hyperparameter Optimization (Optuna):** Optuna searches for parameters that maximize CV performance. Many parameters directly control model complexity and act as regularizers. The script already defines search spaces for these. *If overfitting persists, consider narrowing these search spaces towards less complex/more regularized values (see point 3).*
    *   **Feature Selection:** Reducing the number of input features is a very effective way to reduce model complexity and the risk of overfitting. All included methods help:
        *   `feature_importance_filter`: Removes less important features early on.
        *   `correlation_filtering`: Removes redundant features.
        *   Exhaustive/RFE/GA: Explicitly search for smaller, high-performing feature subsets.
    *   **Regularized Models:** Choosing models with built-in regularization (Ridge, Lasso, ElasticNet, SVM, XGBoost, LightGBM, CatBoost) can inherently help prevent overfitting. The script offers these as choices.

2.  **Strengthen Feature Selection (Configuration Examples):**
    *   **Reduce Initial Features via Importance:**
        *   **Goal:** Keep fewer features after the first filtering step.
        *   **Change:** In `FEATURE_CONFIG`, decrease `importance_threshold`.
        *   **Example:** Change `importance_threshold: 0.2` (keeps top 80% important features) to `importance_threshold: 0.1` (keeps top 90%) or even `importance_threshold: 0.4` (keeps top 60%). *Note: This is a percentile threshold, lower values keep more features, higher values keep fewer.* **Correction:** *Actually, the current implementation uses `1 - importance_threshold` for the percentile, so increasing `importance_threshold` keeps fewer features.* **Example (Corrected):** Change `importance_threshold: 0.2` (uses `np.percentile(..., 100 * (1 - 0.2))`) to `importance_threshold: 0.4` to set a higher importance bar and keep fewer features.
    *   **Reduce Features via Correlation:**
        *   **Goal:** Remove features that are more strongly correlated with others.
        *   **Change:** In `FEATURE_CONFIG`, decrease `correlation_threshold`.
        *   **Example:** Change `correlation_threshold: 0.95` to `correlation_threshold: 0.9` or `correlation_threshold: 0.85`.
    *   **Interpret Advanced Selection Results:**
        *   **RFE:** Examine the `rfe_{metric}_curve.png` plot in the results. Look for the "elbow" point where performance plateaus or worsens as more features are added. If the script automatically chose a point with many features but similar performance to a point with fewer features, you might manually select the feature set with fewer features for the final run (requires minor script modification or rerunning based on the chosen features from `rfe_results.csv`).
        *   **GA/Exhaustive Search:** Look at the `*_results.csv` file. Find the best result (lowest RMSE or highest F1). Then look for results with slightly worse performance but significantly fewer features. You might decide the small performance trade-off is worth the reduced complexity.
        *   **Example:** If the best GA result has 15 features and RMSE 0.5, but another result has 10 features and RMSE 0.52, you might manually choose the 10-feature set for the final model to reduce overfitting risk.

3.  **Adjust Hyperparameter Search Spaces (Configuration Examples):**
    *   **Goal:** Guide Optuna to explore simpler or more regularized models.
    *   **Location:** Modify the `param_space` dictionary within `regression_models` or `classification_models` for the specific model you are using.
    *   **Increase Regularization Strength:**
        *   **Ridge/Lasso/ElasticNet (`alpha`):** Change `(0.0001, 1.0, 'log-uniform')` to `(0.1, 10.0, 'log-uniform')`. (Higher alpha = more regularization).
        *   **XGBoost/LightGBM/CatBoost (`reg_alpha`, `reg_lambda`, `l2_leaf_reg`):** Change lower bounds, e.g., `(0.0, 1.0)` to `(0.1, 5.0)`. (Higher values = more regularization).
        *   **SVC/LogisticRegression (`C`):** This is the *inverse* of regularization strength. Change `(0.1, 100, 'log-uniform')` to `(0.01, 10.0, 'log-uniform')`. (Lower C = more regularization).
    *   **Limit Model Complexity (Trees/Boosting):**
        *   **`max_depth`:** Change `(3, 10)` to `(3, 6)` or `(2, 5)`. (Lower max depth = simpler trees).
        *   **`min_samples_split` / `min_samples_leaf` / `min_child_weight` / `min_child_samples`:** Change `(2, 20)` to `(10, 30)` or `(1, 10)` to `(5, 25)`. (Higher values prevent splitting small nodes, reducing noise fitting).
        *   **`n_estimators` (Forests/Boosting):** Change `(50, 200)` to `(30, 100)`. (Fewer trees can sometimes help, though often counteracted by early stopping or learning rate in boosting).
        *   **`learning_rate` (Boosting):** While counter-intuitive, sometimes *decreasing* the learning rate (`(0.01, 0.3)` to `(0.005, 0.1)`) combined with a *fixed* or slightly increased `n_estimators` and early stopping (if applicable in the model) can lead to better generalization. Optuna handles this search, but narrowing the range can guide it.
    *   **MLP (`alpha`):** Increase regularization, e.g., change `(0.0001, 0.1, 'log-uniform')` to `(0.001, 1.0, 'log-uniform')`. Consider smaller `hidden_layer_sizes`, e.g., change `[(50,), (100,), (50, 25), (100, 50)]` to `[(30,), (50,), (30, 15)]`.

4.  **Enable Gaussian Noise (Configuration Examples):**
    *   **Goal:** Add noise to input features during training to make the model more robust.
    *   **Change:** In `NOISE_CONFIG`:
        *   Set `add_gaussian_noise: True`.
        *   Start with a small `gaussian_noise_scale`, e.g., `gaussian_noise_scale: 0.01` or `gaussian_noise_scale: 0.05`. If overfitting persists, cautiously increase it, e.g., to `0.1`.

5.  **Increase Cross-Validation Folds (Configuration Example):**
    *   **Goal:** Get a more reliable estimate of generalization performance during tuning.
    *   **Change:** In `MODEL_CONFIG`, increase `cv`.
    *   **Example:** Change `cv: 5` to `cv: 10`. (Note: This significantly increases runtime).

6.  **Try Simpler Models:**
    *   **Goal:** Use a model inherently less prone to overfitting.
    *   **Action:** When prompted to "Choose a base model," select options like Linear Regression, Logistic Regression, Ridge, Lasso, or ElasticNet instead of complex models like Gradient Boosting, Random Forest, or MLP.

7.  **Get More Data:** While often not feasible, increasing the amount and *diversity* of your training data is fundamentally the best way to combat overfitting. AutoMatFlow cannot do this, but it's a crucial consideration.

8.  **Iterate:** Addressing overfitting is often an iterative process. Make configuration changes based on the results (especially the train vs. test performance gap), re-run the script, and analyze the new gap.

## Tips for Improving Computation Speed

Automated workflows like this can be computationally intensive. If you find the script running too slowly, consider these adjustments:

1.  **Reduce Hyperparameter Optimization Intensity:**
    *   **Fewer Iterations:** This is often the biggest time saver. Decrease `bayes_iter` in `MODEL_CONFIG`.
        *   **Example:** Change `bayes_iter: 50` to `bayes_iter: 20` or `bayes_iter: 15`. This explores fewer parameter combinations but might find a slightly less optimal set.
    *   **Narrow Search Spaces:** If you have some intuition about good parameter ranges, tighten the bounds in the `param_space` for the chosen model.
        *   **Example (RandomForest):** Instead of `"model__n_estimators": (50, 300)`, use `"model__n_estimators": (50, 150)`.
    *   **Note:** Optuna's `MedianPruner` (used by default) already helps by stopping unpromising trials early.

2.  **Decrease Cross-Validation Folds:**
    *   **Fewer Folds:** Reduce the number of models trained during optimization and evaluation. Decrease `cv` in `MODEL_CONFIG`.
        *   **Example:** Change `cv: 5` to `cv: 3`. This gives a less robust performance estimate but is much faster. Also consider reducing `cv_outer` and `cv_inner` if using nested CV (though nested CV isn't the default path in the current main logic).

3.  **Adjust Feature Selection Strategy:**
    *   **Avoid Exhaustive Search:** This method (`option 1`) is computationally infeasible for more than ~15-20 features after initial filtering. Choose RFE (`option 2`) or Genetic Algorithm (`option 3`) instead for larger feature sets, or skip advanced selection (`option 0`).
    *   **Simplify RFE:** While the script currently evaluates features 1 to `max_features` (default 20), if RFE is too slow, you could modify the code in `rfe_feature_selection` to increase the `step` parameter in `RFE(...)` (e.g., `step=2` removes 2 features per iteration) or reduce the `max_features` variable.
    *   **Tune Genetic Algorithm:** Decrease `genetic_population_size` and/or `genetic_generations` in the main `config`.
        *   **Example:** Change `genetic_population_size: 50` to `20` and `genetic_generations: 30` to `15`. This reduces the search space explored by the GA.
    *   **Reduce Permutation Importance Repeats:** If the initial importance filtering defaults to permutation importance (e.g., for models without direct `.feature_importances_`), it can be slow. Reduce `permutation_repeats` in `MODEL_CONFIG`.
        *   **Example:** Change `permutation_repeats: 5` to `permutation_repeats: 2` or `1`.

4.  **Choose Faster Models:**
    *   **Model Selection:** Some models are inherently faster to train. When prompted, consider selecting:
        *   Linear Models (Linear/Logistic Regression, Ridge, Lasso)
        *   Decision Tree (if depth is limited)
        *   HistGradientBoostingRegressor/Classifier (often very fast)
        *   LightGBM (usually faster than XGBoost/CatBoost, especially on CPU)
    *   **Avoid:** Gaussian Processes and SVM with non-linear kernels tend to be slower, especially on larger datasets. MLP can also be slow depending on architecture and data size. TabPFN has a relatively fixed (and potentially long) inference time.

5.  **Utilize Hardware:**
    *   **CPU Cores (`n_jobs`):** Ensure `n_jobs` in `MODEL_CONFIG` is set to `-1` to use all available CPU cores for parallelizable tasks (like cross-validation within Optuna, some parts of feature selection, ensemble methods). *Be mindful of memory usage when using many cores.* The script attempts to manage parallelism to avoid OS issues, especially on Windows.
    *   **GPU Acceleration (`use_gpu`):** If you have a compatible NVIDIA GPU and the necessary drivers/libraries installed (CUDA, GPU-enabled XGBoost/LGBM/CatBoost/PyTorch), set `use_gpu: True` in `GPU_CONFIG`. This can drastically speed up training for supported models (XGBoost, LightGBM, CatBoost, TabPFN). Check the console output for GPU detection status.

6.  **Reduce SHAP Analysis Scope:**
    *   **Fewer Samples:** Decrease `shap_sample_size` in `SHAP_CONFIG`. KernelExplainer (used for many models) scales poorly with sample size.
        *   **Example:** Change `shap_sample_size: 200` to `shap_sample_size: 50` or `100`.
    *   **Basic Plots Only:** Change `shap_plot_type: "all"` to `shap_plot_type: "basic"` in `SHAP_CONFIG` to skip generating more complex (and slower) dependence and interaction plots.
    *   **Skip SHAP:** If model interpretability is not the primary goal for a specific run, answer 'n' when prompted "是否进行SHAP分析?".

7.  **Data Subsampling (Manual):**
    *   **For Exploration:** If you are just exploring parameters or the workflow, consider running the script initially on a smaller random sample of your data (`df.sample(frac=0.1)` before splitting) to get results faster. Remember to run on the full dataset for final results. (This requires modifying the data loading part of the script).

**Trade-offs:** Remember that most speed improvements involve a trade-off. Reducing iterations, folds, or search spaces might lead to slightly less optimal hyperparameters or feature sets. Choose the balance that fits your needs and available computation time.



## General Tips and Best Practices

Beyond specific issues like overfitting or speed, consider these points for effective use:

1.  **Data Preparation is Key:**
    *   **Before Running:** While the script handles basic imputation and scaling, examine your data beforehand for potential issues like:
        *   **Outliers:** Extreme values can skew results, especially for linear models or distance-based algorithms (KNN). Consider outlier detection and treatment (e.g., capping, removal, transformation) *before* feeding data to the script.
        *   **Skewed Distributions:** Highly skewed numerical features might benefit from transformations (e.g., log, Box-Cox) applied *before* running the script, potentially improving performance for some models. StandardScaler helps center data but doesn't fix skewness fundamentally.
    *   **String Categorical Features:** The script currently handles numerical features and *numerically encoded* categorical features specified in `categorical_num_cols`. **It does not automatically handle string-based categorical features.** If you have columns with text categories (e.g., "Material A", "Region B"), you **must** encode them (e.g., using `pandas.get_dummies` or `sklearn.preprocessing.OrdinalEncoder`) *before* passing the data to `AutoMatFlow.py`. If using one-hot encoding beforehand, ensure you handle potential multicollinearity if needed, though the script's correlation filter might help later.

2.  **Feature Engineering:**
    *   This script focuses on **feature selection**, not feature engineering. It selects the best subset from the features you provide.
    *   Consider creating potentially useful features *before* running the script, based on domain knowledge. This could include:
        *   Interaction terms (e.g., feature A * feature B).
        *   Polynomial features (e.g., feature A squared).
        *   Ratios or differences between features.
        *   Domain-specific calculations (e.g., density from mass and volume).
    *   Adding well-engineered features can sometimes improve model performance more significantly than just selecting from the original set.

3.  **Understanding `categorical_num_cols`:**
    *   This setting is crucial. Use it **only** for columns that contain *numbers* but represent distinct *categories* (e.g., `[1, 2, 3]` representing different process types, not a continuous quantity).
    *   Features listed here will be one-hot encoded after imputation.
    *   If a truly continuous numerical feature is mistakenly listed here, it will be inappropriately one-hot encoded, likely harming performance.
    *   If a numerically encoded categorical feature is *not* listed here, it will be treated as continuous (scaled), which might also be suboptimal depending on the model.

4.  **Interpreting Results Holistically:**
    *   **Focus on Test Set Performance:** While training set scores are reported, the primary goal is generalization. Pay most attention to the **Test Set** metrics in `final_model_performance.csv`.
    *   **Cross-Validation Scores:** Use the CV scores (also in `final_model_performance.csv` and intermediate files) as a measure of model stability and robustness. A large standard deviation in CV scores might indicate sensitivity to data splits.
    *   **Feature Importance vs. Selection:** The initial `feature_importance_results.csv` and SHAP results show feature impact *for the specific model and data split*. The *final selected features* (`top_feature_combinations.csv` or best from RFE/GA) are chosen based on *predictive performance* across CV folds, which might sometimes differ slightly from pure importance rankings.
    *   **SHAP Insights:** Use the SHAP plots not just for importance but to understand *how* features influence predictions (e.g., does higher temperature increase or decrease the target? Is the relationship linear?).

5.  **Reproducibility:**
    *   **`random_state`:** The script uses the `random_state` from `BASIC_CONFIG` in most places (train/test split, models, CV splits, Optuna, noise). Keeping this constant ensures the *same splits and initializations* across runs.
    *   **Environment:** For maximum reproducibility, especially across different machines or time periods, save your exact Python environment using `pip freeze > requirements_exact.txt` after installation and use that specific file for future installations (`pip install -r requirements_exact.txt`). Minor library updates can sometimes cause small numerical differences.
    *   **Parallelism:** Be aware that some parallel operations (especially with many cores) might introduce minor non-determinism, although efforts are made to minimize this.

6.  **Memory Management:**
    *   **Monitor:** Keep an eye on your system's RAM usage during long runs, especially when using `n_jobs = -1`, Exhaustive Search, or large SHAP sample sizes.
    *   **Built-in Checks:** The script includes basic memory checks (`check_memory_usage`) but might not catch all rapid spikes.
    *   **Reduce Load:** If memory is an issue, reduce `n_jobs`, use RFE/GA instead of Exhaustive Search, decrease `shap_sample_size`, or consider running on a machine with more RAM.

7.  **Comparing Base Models:**
    *   The script optimizes and evaluates based on the *single* base model you choose initially.
    *   For a more thorough comparison, consider running the entire `AutoMatFlow.py` workflow *multiple times*, selecting a different promising base model each time (e.g., once with RandomForest, once with XGBoost, once with Ridge). This helps determine which underlying model *type* works best for your specific data *after* feature selection and tuning.

8.  **Incorporate Domain Knowledge:**
    *   **Sanity Check Features:** After the script selects the "best" features, compare this list to your expert understanding of the problem. Are critical known features missing? Are unexpected features included?
    *   **Investigate Discrepancies:** If an important feature was dropped, check its importance score and correlation with other selected features. Perhaps it was redundant or didn't add predictive value *in combination with others*.
    *   **Manual Refinement:** You might use the script's output as a strong suggestion but manually add/remove a feature for a final evaluation based on strong domain reasons.


## Loading the Saved Model

```python
import joblib
import pandas as pd

# Load the saved model and metadata
model_data = joblib.load("path/to/your/results/final_model_with_metadata.pkl")
pipeline = model_data["model"]
selected_features = model_data["feature_names"]

# Load new data (ensure it has the selected feature columns)
new_data = pd.read_csv("new_data.csv")
X_new = new_data[selected_features] # Make sure columns match exactly

# Make predictions
predictions = pipeline.predict(X_new)
# For classification probability:
# probabilities = pipeline.predict_proba(X_new)

print(predictions)