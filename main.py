from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, alexnet

import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PCMU:
    def __init__(self, model, loss_fn, learning_rate, sigma):
        self.loss_fn = loss_fn
        self.sigma = sigma
        self.learning_rate = learning_rate

    def train_one_epoch(self, model, optimizer, scheduler, criterion, data_loader, regular_training=False):
        # Get all the gradients and their average according to Eq.(20)
        model.train()
        gradients = []  # Accumulate gradients for all batches
        avg_gradients = None
        
        for inputs, targets in tqdm(data_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            batch_gradients = torch.cat([param.grad.view(-1) for param in model.parameters()])
            if avg_gradients == None:
                avg_gradients = batch_gradients
            else:
                avg_gradients = self.compute_gradient_average([avg_gradients, batch_gradients])

            #gradients.append(batch_gradients)
            
            if regular_training:
                optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        return avg_gradients

    def compute_gradient_average(self, gradients):
        stacked_tensors = torch.stack(gradients)
        avg_gradients = torch.mean(stacked_tensors, dim=0)
        
        return avg_gradients

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
        noise = torch.normal(mean=0, std=self.sigma, size=quantized_gradients.size()).to(DEVICE)
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
num_epochs = 120
batch_size = 128

#Load the loaders
trainloader, valloader, testloader = datasets.get_loaders('svhn', '../image_data/', 500) 

# Create ResNet-18 model and initialize it with Kaiming
def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
model = resnet18(pretrained=False, num_classes=10)
model.apply(kaiming_init)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(DEVICE)
pcmu = PCMU(model, criterion, learning_rate, sigma)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
scheduler = None#optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# Training loop
for epoch in range(num_epochs):
    # Train for one epoch
    print ("*"*20, f"epoch {epoch}", "*"*20)
    print ("==> Calculate gradients ...")
    avg_gradients = pcmu.train_one_epoch(model, optimizer, scheduler, criterion, trainloader, regular_training=True)
    
    # Calculate the average gradients over the dataset (all the batches)
    print ("==> Average gradients out ...")
    #avg_gradients = pcmu.compute_gradient_average(gradients)
    
    # Apply quantization and smoothing to the average gradients
    print ("==> Quantize gradients ...")
    quantized_gradients = pcmu.quantization_function(avg_gradients)

    print ("==> Gradient smoothing ...")
    smoothed_gradients = pcmu.randomized_gradient_smoothing(quantized_gradients)

    
    # Update the model parameters using the smoothed gradients
    print ("==> Updating parameters ...")
    pcmu.update_parameters(model, smoothed_gradients)

    if epoch == int(num_epochs/2):
        pcmu.learning_rate = pcmu.learning_rate / 10
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / 10
    elif epoch == int(num_epochs*3/4):
        pcmu.learning_rate = pcmu.learning_rate / 10
        for g in optimizer.param_groups:
            g['lr'] = g['lr'] / 10


    # Train set evaluation
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
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
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
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
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print('Test accuracy: {:.2f}%'.format(test_accuracy))