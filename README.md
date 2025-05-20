# Deep Learning: Alphabet Soup Charity Success Predictor

## Table of Contents

- [Background](#background)
- [Technologies Used](#technologies-used)
- [Files](#files)
- [Repository Structure](#repository-structure)
- [Step 1: Data Preprocessing](#step-1-data-preprocessing)
- [Step 2: Compile, Train, and Evaluate the Model](#step-2-compile-train-and-evaluate-the-model)
- [Step 3: Model Optimization](#step-3-model-optimization)
- [Step 4: Summary and Recommendations](#step-4-summary-and-recommendations)

## Background

The nonprofit foundation **Alphabet Soup** seeks to fund organizations that have the highest likelihood of success. To support their selection process, this project develops a **binary classification deep learning model** that predicts whether a funded applicant is likely to be successful, using historical data of over 34,000 organizations.

## Technologies Used

- Python 3.10
- Pandas
- scikit-learn
- TensorFlow/Keras
- Google Colab
- Jupyter Notebook
- HDF5

## Files

- `Starter_Code.ipynb` – Preprocessing, model training, and evaluation
- `AlphabetSoupCharity.h5` – Saved model
- `AlphabetSoupCharity_Optimization.ipynb` – Optimized model for >75% accuracy
- `AlphabetSoupCharity_Optimization.h5` – Saved optimized model

## Repository Structure

```
deep-learning-challenge/
├── output/
│   └── AlphabetSoupCharity.h5
├── .gitignore
├── README.md
└── Starter_Code.ipynb
```

## Step 1: Data Preprocessing

### ✔️ Target and Features

- **Target:** `IS_SUCCESSFUL` – Indicates whether the funded organization was successful.
- **Features:**
  - `APPLICATION_TYPE`
  - `AFFILIATION`
  - `CLASSIFICATION`
  - `USE_CASE`
  - `ORGANIZATION`
  - `STATUS`
  - `INCOME_AMT`
  - `SPECIAL_CONSIDERATIONS`
  - `ASK_AMT`

### ❌ Dropped Columns

- `EIN` – Tax ID (non-informative)
- `NAME` – Organization name (non-informative)

### 📊 Preprocessing Steps

- Combined rare categorical values (e.g., low-frequency `APPLICATION_TYPE` and `CLASSIFICATION`) into a general `"Other"` category.
- Encoded categorical variables using `pd.get_dummies()`.
- Scaled numeric features using `StandardScaler`.
- Split the dataset into training and test sets.

## Step 2: Compile, Train, and Evaluate the Model

### 🔢 Model Architecture

- **Input Features:** ~116 features after one-hot encoding
- **First Hidden Layer:** 80 neurons, ReLU activation
- **Second Hidden Layer:** 30 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation

### ⚙️ Training Configuration

- **Epochs:** 100
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

### 📈 Evaluation

```
Loss: 0.53
Accuracy: 0.73
```

- **Result:** Did not meet 75% target accuracy.

## Step 3: Model Optimization

Three separate attempts were made to optimize performance:

1. **Increased hidden layer size:**
   - Added a third hidden layer with 10 neurons
   - Slight increase in accuracy but still < 75%

2. **Tuned categorical bins:**
   - Adjusted the threshold for rare value replacement (e.g., `APPLICATION_TYPE`, `CLASSIFICATION`)
   - Minor effect on model accuracy

3. **Added dropout layers and changed activation functions:**
   - Slight improvements, but overfitting became a challenge

### ✅ Final Accuracy (Best Optimization Attempt)

```
Loss: 0.49
Accuracy: 0.76
```

- **Result:** Achieved target performance in `AlphabetSoupCharity_Optimization.h5`

## Step 4: Summary and Recommendations

### 🧠 Summary

The optimized neural network was able to meet the target accuracy of over 75%, demonstrating that deep learning can effectively classify potential successful organizations. Careful preprocessing and model tuning were crucial.

### 💡 Recommendation

While the neural network performed well, further improvements may be possible with:

- **Ensemble models** (e.g., Random Forest, Gradient Boosting)
- **Feature engineering** (e.g., combining `USE_CASE` and `ASK_AMT` into new metrics)
- **Hyperparameter tuning** using automated tools like Keras Tuner