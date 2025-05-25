# Air Quality Forecasting with LSTM

## Overview

This project applies Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) models, to forecast PM2.5 air pollution levels in Beijing. Accurate forecasting of air quality is critical for public health, urban planning, and environmental management.

The model was trained using historical air quality and weather data provided via a Kaggle competition hosted by ALU.

---

##  Problem Statement

The objective is to predict future PM2.5 concentrations based on past air quality and meteorological data using a deep learning approach. The primary goal is to minimize the **Root Mean Squared Error (RMSE)** between the predicted and actual PM2.5 levels.

---

## Project Structure
├── data/
│ ├── train.csv
│ ├── test.csv
│ ├── sample_submission.csv
├── preprocess.py
├── model.py
├── main.py
├── predict_submission.py
├── saved_model.h5
├── results/
