# Explaining Condition-Based Maintenance for Metro Trains Using Ensemble ML with XAI and Hybrid Deep Learning Models

M.Tech Research Project (Stage-II) · Computer Science & Engineering (Data Science & Analytics)
Dr. Vishwanath Karad MIT World Peace University, Pune · Academic Year 2024-25

**Author:** Aniket Singh
**Guide:** Dr. Sarika Bobde

---

## Overview

Metro trains are vital to urban transport, but unexpected failures disrupt service and put passenger safety at risk. This project builds a predictive / condition-based maintenance framework for the **Air Production Unit (APU)** of metro trains, which supports braking and door operations.

It tackles two tasks:

1. **Failure classification** – detecting whether the APU is failing.
2. **Remaining Useful Life (RUL) prediction** – estimating how long the component will keep working.

Key ideas:

- **Ensemble ML** (Stacking, Random Forest, Bagging, XGBoost) for classification and regression.
- **Hybrid deep learning** (TCN-GRU), plus standalone CNN, LSTM, GRU and TCN baselines.
- **Down-sampling** of the large time-series data, retaining over 90% of the original variance to cut compute cost.
- **Explainable AI** (SHAP and LIME) on the ensemble models, giving global and local explanations and practical thresholds for maintenance teams (e.g. oil temperature above ~70 °C, pressure above ~8 bar).

## Key Results

| Task | Best model | Result |
|------|-----------|--------|
| Failure classification | Stacking ensemble | ~99.95% on all evaluation metrics |
| RUL prediction | Hybrid TCN-GRU | R² ≈ 0.92, RMSE ≈ 0.26 |

## Dataset

**MetroPT-3**, a real-world dataset of metro train APU sensor readings from Porto, Portugal (available from the UCI Machine Learning Repository).
Features include TP2, TP3, H1, DV pressure, Reservoirs, Motor Current and Oil Temperature.

> **The dataset is not included in this repository.** Download it from UCI and update the file paths in the notebooks (see *Getting Started*).

## Repository Structure

```
.
├── README.md
├── M.Tech project Dissertation Report.docx      # Full dissertation report
├── Research project Conference Paper.pdf        # GINOTECH 2025 paper
├── Review Conference Paper.pdf                  # ICAET 2025 review paper
├── Research Paper Certificate.pdf               # Conference certificate
├── Review paper certificate.pdf                 # Conference certificate
└── Python codes/
    ├── Untitled0.ipynb                # Data loading, exploration & preprocessing (MetroPT-3)
    ├── Project_classification.ipynb   # Ensemble failure classification + SHAP/LIME
    ├── Project_Regression.ipynb       # Ensemble RUL regression + SHAP/LIME
    ├── CNN_model.ipynb                # CNN model for Classification and RUL
    ├── LSTM_model.ipynb               # LSTM model for Classification and RUL
    ├── GRU_Model.ipynb                # GRU model for Classification and RUL
    ├── TCN_model.ipynb                # TCN model for Classification and RUL
    └── Hybrid_model.ipynb             # Hybrid TCN-GRU model for Classification and RUL
```

## Publications

- *Explaining Condition-Based Maintenance for Metro Trains Using Ensemble ML with XAI and Hybrid Deep Learning Models* – 2025 Global Conference in Emerging Technology (GINOTECH), Pune, May 2025. (https://ieeexplore.ieee.org/document/11076995 )
- *A Review on Predicting Failure in Industrial Machines: its Methods, Challenges and Future Direction* – 2025 1st International Conference on AIML-Applications for Engineering & Technology (ICAET), Pune, Jan 2025. (https://ieeexplore.ieee.org/document/10932305 )

## Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · TensorFlow/Keras · SHAP · LIME · Matplotlib · Seaborn · Jupyter / Google Colab

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/singh-aniket123/M.Tech-Final-Year-Project.git
   cd M.Tech-Final-Year-Project
   ```
2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost tensorflow shap lime matplotlib seaborn jupyter
   ```
3. **Get the data** – download MetroPT-3 from UCI and place the CSV(s) locally.
4. **Update file paths** – the notebooks were written for Google Colab and read data from Google Drive, e.g.
   ```python
   df = pd.read_csv('/content/drive/My Drive/2Tresample_data.csv')
   ```
   Change this to your local path, and remove the `google.colab` drive-mount cells if you are not using Colab.
5. **Run the notebooks** in this order: `Untitled0.ipynb` → `Project_classification.ipynb` / `Project_Regression.ipynb` → the deep learning notebooks.

## Future Scope

Extend the approach to other train subsystems and industries, with real-time deployment and adaptive learning.

## Acknowledgements

Thanks to Dr. Sarika Bobde (guide), the expert panel (Dr. Preeti Kale, Dr. Aditi Jahagirdar, Prof. Madhura Phatak) and the Department of Computer Engineering and Technology, MIT World Peace University, Pune.

## License

All rights reserved

## Contact

Aniket Singh – aniketsingh.dpsk@gmail.com
