import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from main import training_data_path, testing_data_path

train_losses = []
train_counter = []

test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(epochs + 1)]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(dim=1, input=x)

def train(ep, net, opt, train_load):
  net.train()
  for batch_idx, (data, target) in enumerate(train_load):
    opt.zero_grad() #Reset Optimizer
    output = net(data) #Feedforward
    loss = F.nll_loss(output, target) #Calculate Loss
    loss.backward() #Backprop
    opt.step() #Take optimizer step
    if batch_idx % log_interval == 0: #Print Log output and save
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        ep, batch_idx * len(data), len(train_load.dataset),
        100. * batch_idx / len(train_load), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((ep-1)*len(train_load.dataset)))
      torch.save(net.state_dict(), 'model.pth')
      torch.save(opt.state_dict(), 'optimizer.pth')

def test(net, test_load):
  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_load:
      output = net(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_load.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_load.dataset),
    100. * correct / len(test_load.dataset)))

