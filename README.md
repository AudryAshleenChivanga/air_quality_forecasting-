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

## Experiment Results

The table below shows the performance of different LSTM model variations submitted to the public leaderboard. The goal was to minimize the RMSE (Root Mean Squared Error) on the PM2.5 predictions.

| Submission File        | Description (Short)           | Public RMSE Score |
|------------------------|--------------------------------|-------------------|
| `subm_fixed1.csv`      | Best performing deep LSTM model | **5147.4140**     |
| `subm_fixed.csv`       | Stacked LSTM with 3 layers      | 5507.6594         |
| `subm_fixed (3).csv`   | Similar to fixed1 with changes  | 5734.1797         |
| `submission.csv`       | First valid baseline model      | 6501.6673         |
| `submission_fixed.csv` | Minor tuning from baseline      | 6581.0497         |
| `submission5.csv`      | Smaller LSTM configuration      | 6539.3920         |
| `submission6.csv`      | Dropout increased               | 6619.5076         |
| `submission9.csv`      | Sequence length adjusted        | 6656.4308         |
| `submission7.csv`      | Added one dense layer           | 6717.1096         |
| `submission8.csv`      | Minor tweaks on submission7     | 6738.9758         |
| `submission4.csv`      | Earlier model with time features| 6858.2507         |
| `submission10.csv`     | Reduced units in LSTM           | 6911.0824         |
| `subm_fixed4.csv`      | Experiment with alternate seed  | 8526.7308         |
| `submission3 (2).csv`  | Poor performance version        | 8644.7615         |
| `submission3 (1).csv`  | Original unoptimized model      | 9616.1108         |
| `submission2.csv`      | Invalid format (Error)          | N/A               |
| `submission2 (1).csv`  | Invalid format (Error)          | N/A               |

``
## Conclusion and Findings

This project explored the use of deep LSTM models for forecasting PM2.5 concentrations in Beijing using time series air quality data. Through extensive experimentation, preprocessing, and model tuning, the following key findings were observed:

- **Deep stacked LSTM models** significantly improved prediction performance by capturing temporal dependencies and sequential patterns in the data.
- **Time-based features** (like hour, day of the week, and month) helped the model learn seasonal and daily trends in pollution levels.
- The **skewed distribution** of PM2.5 values posed challenges, especially for RMSE-based evaluation, since rare high values heavily impacted the metric.
- **Dropout and batch normalization** were essential in preventing overfitting and stabilizing the learning process across deeper models.
- **Feature scaling** using MinMaxScaler ensured that the model trained more efficiently and avoided issues with disparate value ranges.
- Among all the experiments, the best performing model achieved a public RMSE score of **5147.4140**, showing strong predictive ability relative to earlier attempts that scored above 9000.

These results demonstrate that LSTM models, when properly structured and fed with well-engineered features, can effectively forecast air pollution levels. This approach can be applied or extended to other cities or pollution indicators to support real-time air quality monitoring and public health interventions.

## Instructions to Run the Project

To run this project and reproduce the results, follow the steps below:

1. **Clone the Repository**  
   Open a terminal and run:
   ```bash
   git clone https://github.com/AudryAshleenChivanga/air_quality_forecasting-
   ````
   Navigate to the Notebook Folder
After cloning, move into the notebook directory:

````
 
cd /air_quality_forecasting-/notebook
````

Run my notebook in google colab or on kaggle

Happy exploring !