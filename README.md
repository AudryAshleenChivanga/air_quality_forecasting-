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

## Experiment Results with RMSE

## Experiment Results
###  Experiments and RMSE Analysis

We conducted 15 LSTM experiments with varying architectural configurations, including different numbers of LSTM layers, hidden units, dropout rates, and learning rates. The goal was to minimize the **Root Mean Squared Error (RMSE)** on the training data, which measures how well the model's predictions match the actual PM2.5 values.

**Root Mean Squared Error (RMSE) Formula:**
 RMSE is a measure of how far the model's predictions are from the actual values, on average. It is calculated by taking the square root of the average of the squared differences between predicted and actual values.

---

### RMSE Results Table

| Experiment | Configuration (LSTM units)         | Dropout | Learning Rate | Train RMSE |
|------------|-------------------------------------|---------|----------------|-------------|
| Exp_1      | [128, 64, 32]                       | 0.3     | 0.001          | **53.75**   |
| Exp_2      | [64, 64, 32]                        | 0.3     | 0.001          | 54.06       |
| Exp_3      | [128, 64]                           | 0.3     | 0.001          | 54.50       |
| Exp_4      | [128]                               | 0.2     | 0.001          | 55.01       |
| Exp_5      | [64, 32]                            | 0.4     | 0.001          | 55.48       |
| Exp_6      | [128, 128, 64]                      | 0.3     | 0.0005         | 55.58       |
| Exp_7      | [64, 64]                            | 0.2     | 0.001          | 56.67       |
| Exp_8      | [128, 64, 32]                       | 0.5     | 0.001          | 57.05       |
| Exp_9      | [128, 64, 32]                       | 0.3     | 0.0001         | 58.94       |
| Exp_10     | [32, 32, 32]                        | 0.3     | 0.001          | 59.12       |
| Exp_11     | [256, 128, 64]                      | 0.3     | 0.001          | 59.60       |
| Exp_12     | [128, 64]                           | 0.4     | 0.001          | 59.60       |
| Exp_13     | [64]                                | 0.3     | 0.001          | 60.35       |
| Exp_14     | [128, 64, 32] (No BatchNorm)        | 0.3     | 0.001          | 60.36       |
wondering about my 15nth , haha it exploded!
---

###  Observations & Insights

- The best performing configuration was **[128, 64, 32] with dropout 0.3 and learning rate 0.001**, achieving the lowest RMSE of **53.75**.
- Deeper models generally performed better, but excessive depth or too large units (e.g., Exp_11) did not guarantee improvement.
- A very low learning rate (e.g., Exp_9 with 0.0001) led to underfitting.
- Batch normalization helped stabilize training; removing it (Exp_14) slightly degraded performance.
- Regularization via dropout was crucial to preventing overfitting, especially in deeper configurations.



## Conclusion and Findings
## Evaluation: RMSE Analysis
RMSE is a measure of how far the model's predictions are from the actual values, on average. It is calculated by taking the square root of the average of the squared differences between predicted and actual values.
### RMSE Trends Across Experiments

We ran 15 LSTM model configurations with varying layer sizes, dropout rates, and learning rates. The model with configuration `[128, 64, 32]`, dropout of `0.3`, and learning rate of `0.001` performed best with a **Train RMSE of 53.75**. Shallower models and those with higher dropout or fewer units generally performed worse, suggesting they could not capture the underlying temporal patterns as effectively.

### Visual Comparisons and Error Analysis

Visual plots comparing predicted PM2.5 values vs. actual values showed that deeper models were better at following the trends, especially around high spikes. However, all models struggled slightly with large peaks due to the skewness of the target distribution. This may have led to **high RMSE values**, as the error is more sensitive to large deviations.

### Overfitting and Underfitting

Models with too many layers or too little dropout began to overfit the training data — learning the noise instead of the signal. This was mitigated using **Dropout layers** and **Batch Normalization**, which helped stabilize learning and reduce generalization error. On the other hand, models with few units or excessive dropout underfit the data and performed poorly on both training and test sets.

### RNN Challenges: Vanishing/Exploding Gradients

Traditional RNNs often face issues like **vanishing or exploding gradients**, especially with longer sequences. LSTM networks are designed to mitigate this with **gated mechanisms** that help preserve gradients during backpropagation. Additionally, we used the **`tanh` activation**, which is well-suited for LSTMs, and **Batch Normalization** to further stabilize training and avoid gradient issues.

Overall, thoughtful architecture design, proper preprocessing, and an understanding of sequential learning challenges contributed significantly to minimizing RMSE and improving model performance.

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
