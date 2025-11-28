⚡ Electricity Consumption Prediction (CNN-LSTM)

This repository contains a short-term (5-minute interval) electricity demand forecasting model built using a multivariate CNN-LSTM and an optimized Stacked LSTM architecture.
The model is trained on a 4-year (2021–2024) multivariate dataset and achieves high predictive accuracy on real unseen data.

🚀 Key Features

-> Multivariate time-series forecasting

-> Hybrid CNN-LSTM + Stacked LSTM deep learning models

-> Handles nonlinear, volatile energy consumption patterns

-> Sliding window input (30 time steps × features)

-> Strong regularization: L2, Dropout, Early Stopping

-> Clean, modular, reproducible code structure


📊 Model Performance
Metric	Value

R² Score	0.9520

MAPE	4.66%

MAE	199.17 kW

RMSE	302.21 kW


🗄️ Dataset

393,440 rows of 5-minute interval data (Jan 2021–Dec 2024)

Features include:

Power demand

Temperature, dew point, humidity

Wind speed, wind direction

Pressure and engineered lag/time features



🏗️ Final Model Architecture (Optimized)

-> LSTM (150 units, ReLU, return_sequences=True)

-> Dropout (0.2)

-> LSTM (100 units, ReLU)

-> Dropout (0.2)

-> Dense output layer


⚠️ Limitations

Predicts only the next 5 minutes

Deterministic (no uncertainty intervals)

Dataset not region-specific

Limited hyperparameter search due to compute cost


🔭 Future Scope

Multi-step forecasting (30–60 minutes)

Transformer / Attention-based models

Probabilistic forecasting

Real-time smart-grid deployment
