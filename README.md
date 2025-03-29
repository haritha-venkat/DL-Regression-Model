# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: haritha shree.V

### Register Number: 212222230046

```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer


```

### Dataset Information

![Screenshot 2025-03-27 141552](https://github.com/user-attachments/assets/dd7f44d8-350a-4a4d-a101-aad715b57448)

### OUTPUT

![Screenshot 2025-03-27 141557](https://github.com/user-attachments/assets/15392250-1166-4817-9222-4b6b85ef3d25)

![Screenshot 2025-03-27 141604](https://github.com/user-attachments/assets/e1baf560-5fb2-4692-9cd3-fbf419f7d393)



## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
