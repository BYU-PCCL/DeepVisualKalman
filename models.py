import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, train_shapes, validation_shapes, test_shapes):
        super(Net, self).__init__()

        sequence_length, channels, height, width = train_shapes[0]
        sequence_length, state = train_shapes[1]

        self.conv1 = nn.Conv2d(channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(5320, 50)
        self.fc2 = nn.Linear(50, state)

    def forward(self, x):


        # image 1, image 2
        # output is n dimensional
        # h matrix to transform from sensor to state space
        # ekf
        # R, Q

        x = 2 * (x[:, 0].float() / 255.0 - .5)

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.max_pool2d(x, 4)
        x = x.view(-1, 5320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def loss(self, output, target):
        print(output.size(), target[:, 0].size())
        loss = F.nll_loss(output, target[:, 0])
        # log likelihood of output being in distribution P

        pred = output.data.max(1, keepdim=True)[1]
        accuracy = pred.eq(target.data.view_as(pred)).cpu().sum() / float(len(target))
        return loss, {'loss': loss, 'accuracy': accuracy * 100}