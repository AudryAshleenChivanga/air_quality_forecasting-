## Problem

The goal of this project is to forecast **PM2.5 values** in Beijing. PM2.5 (particulate matter with a diameter of less than 2.5 micrometers) includes pollutants like dust, soot, smoke, and liquid droplets that are invisible to the naked eye and pose serious health hazards. Predicting these concentrations can help mitigate risks, support public health decisions, and enable proactive environmental control.

---

## Approach

To solve this problem, we employ **LSTM (Long Short-Term Memory)** models, which are well-suited for handling sequential and time series data. The data undergoes preprocessing, augmentation, and feature engineering. The objective is to build a model that minimizes the **Root Mean Squared Error (RMSE)** when predicting PM2.5 levels.

---

## Data Exploration

1. **Structure**: The dataset contains hourly weather and air quality data. Many numerical columns appeared standardized (e.g., values like -1.58 suggest z-score normalization).

2. **Missing Values**: The target variable `pm2.5` contained missing values. These were imputed using the mean strategy to ensure continuity in the target sequence.

3. **Categorical Encoding**: Wind direction was already one-hot encoded into binary columns (e.g., `cbwd_NW`, `cbwd_SE`, `cbwd_cv`). This is helpful for LSTM models as it avoids misleading ordinal relationships.

4. **PM2.5 Distribution**: The PM2.5 values were **right-skewed** with a long tail towards higher values. This is common in pollution data due to occasional high spikes. These rare events significantly impact RMSE, making models appear worse. While logarithmic transformation was considered, it did not yield better results on the leaderboard.

5. **Temporal Features**: Additional time-based features were extracted from the datetime index—such as `hour`, `dayofweek`, `month`, and `is_weekend`. These helped the LSTM understand temporal patterns like daily or seasonal trends.

6. **Feature Selection**: The features were carefully selected by excluding the target `pm2.5` and unrelated columns like `No`.

7. **Feature Scaling**: Features were scaled using `MinMaxScaler` between 0 and 1. `fit_transform()` was applied on training data and `transform()` on test data to prevent data leakage.

8. **Data Reshaping**: LSTM models expect input in 3D shape: `(samples, timesteps, features)`. The data was reshaped accordingly, with 1 timestep per sample.

---

## Model Design

A **deep stacked LSTM** was used to model the time series. Among 15 experiments, the best-performing model had the following architecture:

- **LSTM Layer 1**: 128 units, `tanh` activation, `return_sequences=True`. Large capacity to learn complex patterns. BatchNormalization for stability and Dropout(0.3) to prevent overfitting.
- **LSTM Layer 2**: 64 units, `tanh`, `return_sequences=True`. Reduced complexity, continues sequence learning.
- **LSTM Layer 3**: 32 units, `tanh`, no return sequences. Final summarizing layer, with Dropout(0.2).
- **Dense Layer**: 16 units, `relu` activation. Transforms LSTM output to a more suitable representation.
- **Output Layer**: Dense(1). Predicts the final PM2.5 value.

The model is compiled with the **Adam optimizer** (learning rate 0.001), **MSE loss**, and **RMSE as a performance metric**.

---

## Experiment Results

| Experiment | Architecture Description           | Train RMSE |
|------------|------------------------------------|------------|
| Exp_1      | 128-64-32 LSTM, Dropout(0.3), Dense | XX.XX      |
| Exp_2      | ...                                | ...        |
| ...        | ...                                | ...        |

> *(Replace with actual results once available)*

---
