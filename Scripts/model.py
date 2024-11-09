import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import training_dataset_loader, testing_dataset_loader

epochs = 3

train_losses = []
train_counter = []

test_losses = []
test_counter = [i*len(training_dataset_loader.dataset) for i in range(epochs + 1)]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.transpose_conv1 = nn.ConvTranspose2d(3, 3, 2, 2)
        self.conv1 = nn.Conv2d(3, 3, 1)
    def forward(self, x):
        #Descent
        previous_conv = []
        for i in range(4):
            x = self.conv(self.conv(x))
            previous_conv.append(x)
            x = F.relu(self.max_pool(x))
        #Bottom
        x = self.conv(self.conv(x))
        #Ascent
        for i in range(4):
            x = self.conv(self.conv(x + previous_conv[-i]))
            x = F.relu(self.transpose_conv1(x))
        #Out
        x = self.conv(self.conv(x))
        x = self.conv1(x)
        return x

    def train(self, epochs, network, optimizer, training_dataset_loader):
        network.train()


# def train(ep, net, opt, train_load):
#   net.train()
#   for batch_idx, (data, target) in enumerate(train_load):
#     opt.zero_grad() #Reset Optimizer
#     output = net(data) #Feedforward
#     loss = F.nll_loss(output, target) #Calculate Loss
#     loss.backward() #Backprop
#     opt.step() #Take optimizer step
#     if batch_idx % log_interval == 0: #Print Log output and save
#       print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         ep, batch_idx * len(data), len(train_load.dataset),
#         100. * batch_idx / len(train_load), loss.item()))
#       train_losses.append(loss.item())
#       train_counter.append(
#         (batch_idx*64) + ((ep-1)*len(train_load.dataset)))
#       torch.save(net.state_dict(), 'model.pth')
#       torch.save(opt.state_dict(), 'optimizer.pth')
#
# def test(net, test_load):
#   net.eval()
#   test_loss = 0
#   correct = 0
#   with torch.no_grad():
#     for data, target in test_load:
#       output = net(data)
#       test_loss += F.nll_loss(output, target, reduction='sum').item()
#       pred = output.data.max(1, keepdim=True)[1]
#       correct += pred.eq(target.data.view_as(pred)).sum()
#   test_loss /= len(test_load.dataset)
#   test_losses.append(test_loss)
#   print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#     test_loss, correct, len(test_load.dataset),
#     100. * correct / len(test_load.dataset)))
#
