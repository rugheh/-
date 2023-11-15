import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from google.colab import drive

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

drive.mount('/content/drive')

torch.manual_seed(777)
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(128*8*16, 625, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )

        self.fc = torch.nn.Linear(625, 2, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc(out)
        return out

learning_rate = 0.003
training_epochs = 15

model = CNN().to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trans = transforms.Compose([transforms.ToTensor()])
train_data = dsets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/courses-main/courses-main/AI_Appl/train_data', transform=trans)
data_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

total_batch = len(data_loader)
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

print('Learning Finished!')

trans = transforms.Compose([transforms.Resize((64, 128)), transforms.ToTensor()])
test_data = dsets.ImageFolder(root='/content/drive/MyDrive/Colab Notebooks/courses-main/courses-main/AI_Appl/test_data', transform=trans)
test_set = DataLoader(dataset=test_data, batch_size=len(test_data))

with torch.no_grad():
    for X_test, Y_test in test_set:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
