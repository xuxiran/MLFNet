import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import config as cfg
import torch.autograd as autograd


class Dft(nn.Module):
    def __init__(self, device, decision_window):
        super(Dft, self).__init__()
        self.model = base(decision_window)
        self.drop_f = 2/3*100
        self.drop_b = 2/3*100

        self.num_classes = 2
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    def train(self, train_loader, device, epoch, N_epoch):
        self.model.train()
        count = 0
        all_correct = 0
        for iter, (X,label,identities) in enumerate(tqdm.tqdm(train_loader, position=0, leave=True), start=1):

            all_y = label.to(device)
            all_o = torch.nn.functional.one_hot(all_y, self.num_classes)
            all_f, all_p = self.model(X[2].to(device),X[3][:,5:10,:,:].to(device))

            # Equation (1): compute gradients with respect to representation
            all_g = autograd.grad((all_p * all_o).sum(), all_f)[0]

            # Equation (2): compute top-gradient-percentile mask
            percentiles = np.percentile(all_g.cpu(), self.drop_f, axis=1)
            percentiles = torch.Tensor(percentiles)
            percentiles = percentiles.unsqueeze(1).repeat(1, all_g.size(1))
            mask_f = all_g.lt(percentiles.to(device)).float()

            # Equation (3): mute top-gradient-percentile activations
            all_f_muted = all_f * mask_f

            # Equation (4): compute muted predictions
            all_p_muted = self.model.g_net(all_f_muted)

            # Section 3.3: Batch Percentage
            all_s = F.softmax(all_p, dim=1)
            all_s_muted = F.softmax(all_p_muted, dim=1)
            changes = (all_s * all_o).sum(1) - (all_s_muted * all_o).sum(1)
            percentile = np.percentile(changes.detach().cpu(), self.drop_b)
            mask_b = changes.lt(percentile).float().view(-1, 1)
            mask = torch.logical_or(mask_f, mask_b).float()

            # Equations (3) and (4) again, this time mutting over examples
            all_p_muted_again = self.model.g_net(all_f * mask)

            # Equation (5): update
            loss = F.cross_entropy(all_p_muted_again, all_y)
            count += all_p_muted_again.shape[0]
            pred = torch.max(all_p_muted_again, 1)[1]
            correct = (pred == all_y).sum().item()
            all_correct += correct

            # loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        print(f"Dft Train Accuracy: {all_correct/count}")


    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X,y,z) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                _, convScore = self.model(X[2].to(device),X[3][:,5:10,:,:].to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.norm(X)
        y = self.conv(y)
        y = self.relu(y)
        return y

class DenseBlock(nn.Module):

    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        in_c = in_channels
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.conv1 = ConvBlock(in_c, out_channels)
        in_c = in_c + out_channels
        self.conv2 = ConvBlock(in_c, out_channels)
        in_c = in_c + out_channels
        self.conv3 = ConvBlock(in_c, out_channels)
        in_c = in_c + out_channels
        self.conv4 = ConvBlock(in_c, out_channels)

        self.out_channels = in_channels + num_convs * out_channels # get the out channels

    def forward(self, X):
        y1 = self.conv1(X)
        X = torch.cat((X, y1), dim=1)
        y2 = self.conv2(X)
        X = torch.cat((X, y2), dim=1)
        y3 = self.conv3(X)
        X = torch.cat((X, y3), dim=1)
        y4 = self.conv4(X)
        y = torch.cat((X, y4), dim=1)

        return y


class TransitionBlock_tem(nn.Module):
    def __init__(self, in_channels, out_channels,pool_size = 7):
        super(TransitionBlock_tem, self).__init__()
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        # self.pool = nn.AvgPool3d(kernel_size=(pool_size, 1, 1), stride=(max(1,pool_size//2), 1, 1))

    def forward(self, X):
        y = self.norm(X)
        y = self.relu(y)
        y = self.conv(y)
        # y = self.pool(y)

        return y

class TransitionBlock_fre(nn.Module):
    def __init__(self, in_channels, out_channels,pool_size = 7):
        super(TransitionBlock_fre, self).__init__()
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        self.relu = nn.ReLU()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        # self.pool = nn.AvgPool3d(kernel_size=(pool_size, 1, 1), stride=(max(1,pool_size//2), 1, 1))

    def forward(self, X):
        y = self.norm(X)
        y = self.relu(y)
        y = self.conv(y)
        # y = self.pool(y)

        return y

class DenseNet_3D_t(nn.Module):
    def __init__(self,decision_window):
        super(DenseNet_3D_t, self).__init__()
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        # self.fc = nn.Linear(in_features=1*10*11, out_features=64)
        self.sigmoid = nn.Sigmoid()

        num_channels = 2

        # self.conv1 = nn.Conv3d(10, num_channels, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))

        num_convs = 4

        growth_rate = 5
        self.DB1 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB1.out_channels
        self.TB1 = TransitionBlock_tem(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 1

        self.DB2 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB2.out_channels
        self.TB2 = TransitionBlock_tem(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 1

        self.DB3 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB3.out_channels
        self.TB3 = TransitionBlock_tem(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 1

        self.DB4 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB4.out_channels
        self.TB4 = TransitionBlock_tem(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 1

        self.avgpool = nn.AvgPool3d(kernel_size=(decision_window, 1, 1), stride=1)
        self.cnn = nn.Conv3d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        # Layer 1
        x = x.unsqueeze(dim=1)
        x = self.norm(x)
        eeg_raw = x


        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB1(x)
        x = self.TB1(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB2(x)
        x = self.TB2(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB3(x)
        x = self.TB3(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB4(x)
        x = self.TB4(x)

        # x = self.sigmoid(x)
        x = torch.cat((x, eeg_raw), dim=1)
        x = self.cnn(x)
        x = self.sigmoid(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x = self.fc(x)


        return x

class DenseNet_3D_f(nn.Module):
    def __init__(self,decision_window):
        super(DenseNet_3D_f, self).__init__()
        self.norm = nn.LayerNorm([10,11], elementwise_affine=False)
        # self.fc = nn.Linear(in_features=1*10*11, out_features=64)
        self.sigmoid = nn.Sigmoid()

        num_channels = 10

        # self.conv1 = nn.Conv3d(10, num_channels, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1))

        num_convs = 4

        growth_rate = 5
        self.DB1 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB1.out_channels
        self.TB1 = TransitionBlock_fre(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.DB2 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB2.out_channels
        self.TB2 = TransitionBlock_fre(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.DB3 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB3.out_channels
        self.TB3 = TransitionBlock_fre(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.DB4 = DenseBlock(num_convs, num_channels, growth_rate)
        num_channels = self.DB4.out_channels
        self.TB4 = TransitionBlock_fre(num_channels, num_channels // 2)
        num_channels = num_channels // 2 + 5

        self.avgpool = nn.AvgPool3d(kernel_size=(decision_window, 1, 1), stride=1)
        self.cnn = nn.Conv3d(num_channels, 1, kernel_size=1)

    def forward(self, x):
        # Layer 1
        x = x.unsqueeze(dim=2)
        x = self.norm(x)
        eeg_raw = x


        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB1(x)
        x = self.TB1(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB2(x)
        x = self.TB2(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB3(x)
        x = self.TB3(x)

        x = torch.cat((x, eeg_raw), dim=1)
        x = self.DB4(x)
        x = self.TB4(x)

        # x = self.sigmoid(x)
        x = torch.cat((x, eeg_raw), dim=1)
        x = self.cnn(x)
        x = self.sigmoid(x)
        # x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x = self.fc(x)


        return x




class GNet(nn.Module):
    def __init__(self, N, dim):
        super(GNet, self).__init__()
        self.linear = nn.Linear(dim, N)

    def forward(self, x):
        x = self.linear(x)
        return x

class FeatureCNN(nn.Module):
    def __init__(self, decision_window):
        super(FeatureCNN, self).__init__()
        self.tcnn = DenseNet_3D_t(decision_window)
        self.fcnn = DenseNet_3D_f(decision_window)
        # self.fc = nn.Linear(128, 64)

    def forward(self, x1,x2):

        x1 = self.tcnn(x1)
        x2 = self.fcnn(x2)
        x = torch.cat((x1, x2), dim=1)


        return x



class base(nn.Module):
    def __init__(self, decision_window):
        super(base, self).__init__()
        self.feature_cnn = FeatureCNN(decision_window)
        self.g_net = GNet(2, 220)

    def forward(self, x1,x2):

        x = self.feature_cnn(x1,x2)
        out = self.g_net(x)
        return x,out



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 64
    model = base(decision_window).to(device)

    x1 = torch.rand((10, decision_window,10,11)).to(device)
    x2 = torch.rand((10, 5,10,11)).to(device)

    out = model(x1,x2)
    # summary(model,(5,10,11))
