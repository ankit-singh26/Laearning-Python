# Gradient Descent with Manual Weight Updates
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Number of samples
num_samples = 1000

# Generate random input features (2D)
x = torch.randn(num_samples, 2)

# True weights and bias
true_weights = torch.tensor([1.3, -1])
true_bias = torch.tensor([-3.5])

# Generate target variable
y = x @ true_weights + true_bias

# Plot the dataset
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(x[:, 0], y)
ax[1].scatter(x[:, 1], y)

ax[0].set_xlabel('X1')
ax[0].set_ylabel('Y')
ax[1].set_xlabel('X2')
ax[1].set_ylabel('Y')
plt.show()

# Define the Linear Regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# Define model dimensions
input_size = x.shape[1]
output_size = 1

# Instantiate the model
model = LinearRegression(input_size, output_size)

# Manually set random weights and biases
with torch.no_grad():
    model.linear.weight.copy_(torch.randn(1, input_size))
    model.linear.bias.copy_(torch.rand(1))

# Print initial parameters
weight, bias = model.parameters()
print('Initial Weight:', weight)
print('Initial Bias:', bias)

# Define Mean Squared Error loss
def Mean_Squared_Error(prediction, actual):
    return ((actual - prediction.squeeze()) ** 2).mean()

# Check initial loss
y_p = model(x)
loss = Mean_Squared_Error(y_p, y)
print('Initial Loss:', loss.item())

# Training configuration
num_epochs = 1000
learning_rate = 0.01

# Subplot for weight & bias vs loss
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_p = model(x)
    loss = Mean_Squared_Error(y_p, y)
    
    # Backward pass
    loss.backward()

    # Update weights and biases manually
    with torch.no_grad():
        model.linear.weight -= learning_rate * model.linear.weight.grad
        model.linear.bias -= learning_rate * model.linear.bias.grad

        model.linear.weight.grad.zero_()
        model.linear.bias.grad.zero_()
    
    # Plotting and logging
    if (epoch + 1) % 100 == 0:
        ax1.plot(model.linear.weight.detach().numpy()[0], [loss.item()] * input_size, 'r*-')
        ax2.plot(model.linear.bias.detach().numpy(), loss.item(), 'g+-')
        print(f'Epoch [{epoch+1}/{num_epochs}], Weight: {model.linear.weight.detach().numpy()}, Bias: {model.linear.bias.detach().numpy()}, Loss: {loss.item():.4f}')

# Final plot labels
ax1.set_xlabel('Weight')
ax2.set_xlabel('Bias')
ax1.set_ylabel('Loss')
ax2.set_ylabel('Loss')
plt.show()

# Print final learned weights and bias
w = model.linear.weight
b = model.linear.bias
print(f'\nFinal Learned Parameters:\nWeight = {w.detach().numpy()}\nBias = {b.detach().numpy()}')
