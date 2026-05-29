# ML Research Notebooks

Jupyter notebooks for ML experiments and assignment reporting. Each assignment has its own subdirectory (`a1/`, `a2/`, ...).

## What Has Been Tried

| Assignment | Topic | Notes |
| ------------ | ------- | ------- |
| $A1$ | Project setup | Setup of dagshub and utils file |
| $A2$ | Regression model setup | Setup baseline for A2 with linear regression, removed outliers using cooks distance. Threshold multiplier 3 was the best. |
| $A3$ | Weak-Link classification setup | Tried RFC, Boosting, Logreg, and ETC. "Best" model from Boosting. |
| $A4$ | Comparison of models and improving regreesion model | Implemented t-test to compare models into utils file aswell as improving A2 using Lasso. |
| $A5$ | Ensemble implementation in regression and weak link | Implemented stacking ensembles to both to which improved results |
| $A5b$ | Hyperparameter tuning | Tested different hyperparameters to try and improve results |
| $A6$ | SVM grid optimization | Weaklink level0 implementation of new optimized parameters |
| $A7$ | PCA and Clustering | Did PCA and clustering on weak link data |
| $A8$ | Preparation for DL | Prepared for DL by reading docs and tutorials |
| $A9$ | DeepLearning model to predict z | Using kinectdata transformed to mediapipe to train a model to predict z form x,y |
| $A10$ | Z-predictor model improvements | Fixed data leakage (file-level train/val/test splits), applied y-axis mirroring augmentation, increased hidden size to 256; avg joint error improved from ~3.24 cm to ~2.64 cm |
| $A11$ | LSTM squat activity classification | Trained bidirectional LSTM to classify squat segments (cut vs. uncut); 25 experiments on DagsHub covering optimizers, layer counts, sequence lengths, dropout, and feature scaling; 10-fold CV |
| $A12$ | Session analysis pipeline | Integrated Start_Stop_Predictor_ModelV2 for per-frame exercise detection; end-to-end analyze-session endpoint; frontend switched from real-time streaming to record-and-send |
| $A13$ | Good vs Bad Exercise Classifier | Trained `GoodBad_ClassifierV2` using the A12 Start/Stop model to segment sequences; augmentation (mirror, rotate, scale, noise); feature engineering with 16 inter-joint distances + 6 joint angles; fixed-length resampling to 10 frames |
| $A14$ | Good vs Ugly Exercise Classifier | This was not implementedwith a model but rather with a Rule-based approach |
| $A15$ | Squat Exercise Scoring Classifier | - |
| $A16$ | - | - |

Models are tracked in MLflow on DagsHub using a utils file, ml_utils.py.

### Models

A2 (Regression Model) - Regression model to predict aimoscore
A3 (Weaklink Model) - Model that predicts th weakest link
A9 (Z Prediction Model) - Model that predicts z from x,y (Unused to in application to clear up response times)
A13 (Good vs Bad Classifier) - Model that classifies good vs bad exercise
A15 (Squat Scoring Classifier) - Model that scores squat exercises

## Setup

### Prerequisites

- Python 3.9.12
- Jupyter Notebook or JupyterLab
