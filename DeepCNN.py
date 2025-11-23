#Alright the first thing you have to know is that Deep learning is kinda the Tesla of AI algorithm...But it requires some mildly heavy pc resources(depends on how heavy your dataset, your custom model...etc)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans

#Remember that I'm using pytorch for deep learning examples...but you can also use tensorflow/keras as well

#Setting a random seed for reproducibility(This step will be repeated)
np.random.seed(42)
torch.manual_seed(42)

#deep learning: computer vision(CV)
#algorithm: Simple Convolutional Neural Network(CNN)(PyTorch)
#field: Image Recognition (e.g., object detection, classification)

class SimpleCNN(nn.Module):
    """
    A basic CNN model for image classification, demonstrating the core CNN architecture.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        #Convolutional Layer: Extracts features (edges, textures)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        #Pooling Layer: Reduces dimensionality and makes model translation-invariant
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Fully Connected Layer (Classifier)
        #Input size calculation depends on conv/pool operations
        self.fc = nn.Linear(16 * 14 * 14, 10) # 16 filters * 14x14 output size (from 28x28 input)

    def forward(self, x):
        #x shape [batch_size, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))
        #x shape [batch_size, 16, 14, 14]
        
        #Flatten the feature maps for the fully connected layer
        x = x.view(-1, 16 * 14 * 14)
        
        #Output is 10 classes (e.g., MNIST digits 0-9)
        x = self.fc(x)
        return x

def run_simple_cnn():
    """
    Sets up and runs a training loop for the SimpleCNN using PyTorch's workflow.
    """
    print("\nDeep Learning: Simple CNN (PyTorch) for Computer Vision")
    
    #Simulate data (e.g., 100 images, 1 channel, 28x28 size, 10 classes)
    mock_data = torch.randn(100, 1, 28, 28)
    mock_labels = torch.randint(0, 10, (100,))

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss() #Standard loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    #Simple Training Loop (1 Epoch)
    print("Starting CNN training (1 Epoch)...")
    for i in range(10): #Take 10 steps
        #Zero the gradients
        optimizer.zero_grad()
        
        #Forward pass
        outputs = model(mock_data)
        
        #Calculate Loss
        loss = criterion(outputs, mock_labels)
        
        #Backward pass (Backpropagation: The core DL algorithm)
        loss.backward()
        
        #Update weights
        optimizer.step()
    
    print(f"Loss after 10 steps: {loss.item():.4f}")
    print("CNN training simulated.")
    
#Note In a real scenario, you would use a proper dataset (like MNIST), DataLoader for batching, and multiple epochs.
if __name__ == "__main__":
  run_simple_cnn()
