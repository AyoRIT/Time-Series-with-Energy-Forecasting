# Energy Consumption Forecasting

This project focuses on forecasting energy consumption using RNN-based models. It includes preprocessing steps for time-series data, feature engineering, and implementing a recurrent neural network (RNN) to predict future energy usage based on historical patterns. Key error metrics such as MAE, MAPE, and RMSE are computed to evaluate model performance.

## Requirements

Before running the project, ensure you have the following tools, libraries, and dependencies installed:

### 1. **Environment**
   - Python 3.8 or higher
   - An IDE or code editor (e.g., VSCode, Jupyter Notebook)

### 2. **Libraries**
   Install the required Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preprocessing and Feature Engineering

The following steps were taken to preprocess the data and engineer features for the model:

1. **Feature Selection**: Selected the following columns to include in the dataset:
   - `Hour`, `Year`, `Day of the Month`, `Day of the Year`, `Quarter`, `Energy Consumption`
   - One-hot encoded columns for `Day of the Week` (`Day of the Week_1` to `Day of the Week_6`)
   - One-hot encoded columns for `Month` (`Month_2` to `Month_12`)

2. **One-Hot Encoding**: Converted categorical variables such as `Day of the Week` and `Month` into one-hot encoded columns using `pd.get_dummies()`. Dropped the first category to avoid multicollinearity. i.e. one of the columns can be perfectly predicted from the others, which could negatively affect the machine learning model

3. **Creating sequences**: Reshaped the data into sequences for time-series forecasting. Each sequence contains the past 24 hours of data (24 timesteps), and the corresponding target is the energy consumption of the following hour.

4. **Train-Test Split**: 
   - Used an 80-20 split to divide the dataset into training and testing subsets.
   - Ensured the split maintained the temporal order of the time series to preserve temporal dependencies.

5. **Normalization**: Normalized all input features using a `MinMaxScaler` to ensure they fall within a similar range. This helps the RNN converge faster during training.

6. **Final Dataset Shape**:
   - `X_train` and `X_test`: Shape is `(data_length, 24, 22)` where 24 is the number of timesteps, and 22 is the number of features.
   - `y_train` and `y_test`: Shape is `(data_length, 1)` representing the target energy consumption value.

With this preprocessing, the data is ready for the RNN model implementation.

## Model Architecture

The architecture of the RNN model used in this project is as follows:

 **RNN Model Class**:
   - The model is implemented as a subclass of `torch.nn.Module`.
   - The key components of the model are:
     - `nn.RNN`: The main recurrent neural network layer, which processes the sequential input data.
     - `nn.Linear`: A fully connected layer to map the RNN's output to the target value.

## Model Training and Evaluation

This section details the process of training the RNN model using the training data and evaluating its performance on the validation set.

### 1. **Data Preparation for Training**:
   - The data is split into training and validation sets using a time series-aware split.
   - Each data split is converted into PyTorch tensors and loaded into a `DataLoader` for batch processing.

### 2. **Training Loop**:
   - The training loop iterates over the number of epochs and batches in the training data.
   - For each batch:
     1. Forward pass: Compute the model's predictions.
     2. Loss computation: Use Mean Squared Error (MSE) as the loss function.
     3. Backward pass: Compute gradients and update model parameters using the Adam optimizer.

### 3. **Validation Loop**:
   - After each epoch, the model is evaluated on the validation set.
   - The validation loss is computed to track model performance and guide hyperparameter tuning.

## Model Testing and Error Metrics

This step focuses on using the trained model to make predictions on the test set and compute relevant error metrics to evaluate its performance.

### 1. **Test Data Preparation**:
   - Reshape the test set to match the RNN input requirements: `(batch_size, sequence_length, input_size)`.
   - Convert the test set to PyTorch tensors.

### 2. **Prediction**:
   - Use the trained model to predict energy consumption values for the test set.
   - Ensure the model is in evaluation mode (`model.eval()`) during this step to disable dropout and batch normalization layers (if any).

### 3. **Inverse Scaling**:
   - Transform the predicted and actual values back to their original scale using the `scaler` that was fit during preprocessing.

### 4. **Error Metrics**:
   - Compute the following metrics to evaluate model performance:
     1. **Mean Absolute Error (MAE)**: Measures average absolute difference between actual and predicted values.
     2. **Mean Absolute Percentage Error (MAPE)**: Represents prediction accuracy as a percentage.
     3. **Root Mean Squared Error (RMSE)**: Penalizes large prediction errors more heavily.
