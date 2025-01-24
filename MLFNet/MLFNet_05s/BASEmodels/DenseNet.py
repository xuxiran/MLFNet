import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import config as cfg
from torchsummary import summary

class DenseNet(nn.Module):
    def __init__(self, device, decision_window):
        super(DenseNet, self).__init__()
        self.model = DenseNet_3D(decision_window)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        count = 0
        all_correct = 0
        for iter, (X, y, z) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):

            convScore = self.model(X[2].to(device))
            y = y.to(device)
            count += convScore.shape[0]
            loss = nn.CrossEntropyLoss()(convScore, y)

            pred = torch.max(convScore, 1)[1]
            correct = (pred == y).sum().item()
            all_correct += correct

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"DenseNet Train Accuracy: {all_correct/count}")

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X, y, z) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                out = self.model(X[2].to(device))
                result = np.append(result, torch.max(out, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt


class DenseBlock(nn.Module):
    def conv_block(self,in_channels, out_channels):
        blk = nn.Sequential(nn.BatchNorm3d(in_channels),
                            nn.ReLU(),
                            nn.Conv3d(in_channels, out_channels, kernel_size=(1,3,3), padding=(0,1,1)))
        return blk

    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        net = []
        for i in range(num_convs):
            in_c = in_channels + i * out_channels

            # padding
            # pad_time = nn.ReplicationPad3d((0, 0, 0, 1, 1))
            # net.append(pad_time)
            net.append(self.conv_block(in_c, out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs * out_channels # get the out channels

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)  # concat the input dim and output dim
        return X



class DenseNet_3D(nn.Module):
    def __init__(self,decision_window):
        super(DenseNet_3D, self).__init__()
        self.num_channels = 64
        self.growth_rate = 32

        self.decision_window = decision_window
        self.feature = self.densenet(self.decision_window)
        self.linear = nn.Linear(248, 2)


    def transition_block(self, in_channels, out_channels,avgconv = 7):
        blk = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)),
            nn.AvgPool3d(kernel_size=(avgconv, 2, 2), stride=(3, 1, 1))
        )
        return blk

    def densenet(self,decision_window):
        net = nn.Sequential(
            nn.Conv3d(1, self.num_channels, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(self.num_channels),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0, 1, 1))
        )

        num_channels, growth_rate = self.num_channels, self.growth_rate  # num_channels is the currenct channels
        num_convs_in_dense_blocks = [4,4,4,4]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            DB = DenseBlock(num_convs, num_channels, growth_rate)
            net.add_module("DenseBlosk_%d" % i, DB)
            # last channel
            num_channels = DB.out_channels
            # reduce the output channel
            if i != len(num_convs_in_dense_blocks) - 1:
                if decision_window == 64:
                    net.add_module("transition_block_%d" % i, self.transition_block(num_channels, num_channels // 2,5))
                else:
                    net.add_module("transition_block_%d" % i, self.transition_block(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        net.add_module("BN", nn.BatchNorm3d(num_channels))
        net.add_module("relu", nn.ReLU())
        return net


    def forward(self, x):
        # Layer 1
        x = x.unsqueeze(dim=1)

        x = self.feature(x)
        x = F.avg_pool3d(x, kernel_size=x.size()[2:])
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 256
    model = DenseNet_3D(decision_window).to(device)
    x_input = torch.randn(1, decision_window,10,11).to(device)
    summary(model, (decision_window,10,11))
    out = model(x_input)
    print(out)