from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, alexnet


class PCMU:
    def __init__(self, model, loss_fn, learning_rate, sigma):
        self.loss_fn = loss_fn
        self.sigma = sigma
        self.learning_rate = learning_rate

    def train_one_epoch(self, model, optimizer, scheduler, criterion, data_loader, regular_training=False):
        # Get all the gradients and their average according to Eq.(20)
        model.train()
        gradients = []  # Accumulate gradients for all batches
        
        for inputs, targets in tqdm(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            batch_gradients = torch.cat([param.grad.view(-1) for param in model.parameters()])

            gradients.append(batch_gradients)
            
            if regular_training:
                optimizer.step()
        
        scheduler.step()

        return gradients

    def compute_gradient_average(self, gradients):
        flattened_gradients = torch.cat([grad.view(-1) for grad in gradients])
        avg_gradient = flattened_gradients.mean(dim=0)
        return avg_gradient

    def quantization_function(self, gradients):
        # Simulated quantization function based on Eq.(6)
        sigma = self.sigma 
        quantized_gradients = torch.softmax(
            torch.stack(
                [
                    -(gradients - sigma ** 2),
                    -gradients,
                    -(gradients + sigma ** 2)
                ],
                dim=1
            ),
            dim=1
        )
        return quantized_gradients

    def randomized_gradient_smoothing(self, quantized_gradients):
        # Simulated smoothing function based on Eq.(21)
        noise = torch.normal(mean=0, std=self.sigma, size=quantized_gradients.size())
        smoothed_gradients = torch.argmax(quantized_gradients + noise, dim=-1)
        return smoothed_gradients

    def update_parameters(self, model, smoothed_gradients):
        # Updating the params based on Eq.(29)
        start = 0
        for param in model.parameters():
            end = start + param.numel()
            param.data -= self.learning_rate * smoothed_gradients[start:end].reshape(param.shape)
            start = end

# Example usage
learning_rate = 0.05
sigma = 0.1
num_epochs = 26

# Load CIFAR-10 dataset

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../image_data/cifar10/', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='../image_data/cifar10/', train=False, download=True, transform=transform)

# Split trainset into train and validation sets
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size
trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=500, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False, num_workers=2)

# Create ResNet-18 model
model = resnet18(pretrained=False, num_classes=10)


criterion = nn.CrossEntropyLoss()
pcmu = PCMU(model, criterion, learning_rate, sigma)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# Training loop

for epoch in range(num_epochs):
    # Train for one epoch
    print ("*"*20, f"epoch {epoch}", "*"*20)
    print ("==> Calculate gradients ...")
    gradients = pcmu.train_one_epoch(model, optimizer, scheduler, criterion, trainloader, regular_training=True)
    
    # Calculate the average gradients over the dataset (all the batches)
    print ("==> Average gradients out ...")
    stacked_tensors = torch.stack(gradients)
    avg_gradients = torch.mean(stacked_tensors, dim=0)
    
    # Apply quantization and smoothing to the average gradients
    print ("==> Quantize gradients ...")
    quantized_gradients = pcmu.quantization_function(avg_gradients)

    print ("==> Gradient smoothing ...")
    smoothed_gradients = pcmu.randomized_gradient_smoothing(quantized_gradients)

    
    # Update the model parameters using the smoothed gradients
    print ("==> Updating parameters ...")
    pcmu.update_parameters(model, smoothed_gradients)


    # Train set evaluation
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
    train_accuracy = 100 * train_correct / train_total

    # Validation set evaluation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_accuracy = 100 * val_correct / val_total

    print('Train accuracy: {:.2f}%'.format(train_accuracy), '\tValidation accuracy: {:.2f}%'.format(val_accuracy))

# Final test set evaluation
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print('Test accuracy: {:.2f}%'.format(test_accuracy))