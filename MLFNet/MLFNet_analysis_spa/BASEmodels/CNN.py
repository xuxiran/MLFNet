import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import config as cfg
from torchsummary import summary
class CNN(nn.Module):
    def __init__(self, device, decision_window):
        super(CNN, self).__init__()
        self.model = cnn(decision_window)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        count = 0
        all_correct = 0
        for iter, (X, y, z) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):

            convScore = self.model(X[0].to(device))
            y = y.to(device)
            count += convScore.shape[0]
            loss = nn.CrossEntropyLoss()(convScore, y)

            pred = torch.max(convScore, 1)[1]
            correct = (pred == y).sum().item()
            all_correct += correct

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"CNN Train Accuracy: {all_correct/count}")
    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X, y, z) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                convScore = self.model(X[0].to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt


class cnn(nn.Module):
    def __init__(self, decision_window):
        super(cnn, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17, 64), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(decision_window, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=2)
        self.softmax = nn.Sigmoid()


    def forward(self, x):
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)

        fc_out = self.fc1(flatten_out)
        fc_out = self.softmax(fc_out)
        fc_out = self.fc2(fc_out)

        return fc_out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 64
    model = cnn(decision_window).to(device)
    x = torch.rand((1, decision_window,64)).to(device)
    out = model(x)
    summary(model, (decision_window, 64))
    print(out.shape)
    print(out)

    print('Done')