import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import config as cfg
from torchsummary import summary
class DCNN(nn.Module):
    def __init__(self, device, decision_window):
        super(DCNN, self).__init__()
        self.model = dcnn(decision_window)
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
        print(f"DCNN Train Accuracy: {all_correct/count}")
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


class dcnn(nn.Module):
    def __init__(self, decision_window):
        super(dcnn, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(1,8), stride=(1, 1), padding='same')
        self.norm1 = nn.BatchNorm2d(24)
        self.mish = nn.Mish()
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(1,16), stride=(1, 1), padding='same')
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(1,32), stride=(1, 1), padding='same')

        self.conv4 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(64,1), stride=(1, 1))
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop = nn.Dropout(p=0.2)
        self.norm2 = nn.BatchNorm2d(48)

        self.conv5 = nn.Conv2d(in_channels=48, out_channels=24, kernel_size=(1,16), stride=(1, 1), padding='same')
        self.norm3 = nn.BatchNorm2d(24)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 8))
        mul = 256 // decision_window
        self.fc1 = nn.Linear(in_features=192//mul, out_features=96//mul)
        self.fc2 = nn.Linear(in_features=96//mul, out_features=48//mul)
        self.fc3 = nn.Linear(in_features=48//mul, out_features=24//mul)
        self.fc4 = nn.Linear(in_features=24//mul, out_features=2)


    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = x.permute(0, 1, 3, 2)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.mish(x)

        x = self.conv2(x)
        x = self.norm1(x)
        x = self.mish(x)

        x = self.conv3(x)
        x = self.norm1(x)
        x = self.mish(x)

        x = self.conv4(x)
        x = self.norm2(x)
        x = self.mish(x)

        x = self.pooling1(x)
        x = self.drop(x)

        x = self.conv5(x)
        x = self.norm3(x)
        x = self.mish(x)

        x = self.pooling2(x)
        x = self.drop(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.mish(x)
        x = self.fc2(x)
        x = self.mish(x)
        x = self.fc3(x)
        x = self.mish(x)
        x = self.fc4(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 64
    model = dcnn(decision_window).to(device)
    x = torch.rand((1, decision_window,64)).to(device)
    summary(model, (decision_window, 64))
    out = model(x)
    print(out.shape)
    print(out)

    print('Done')