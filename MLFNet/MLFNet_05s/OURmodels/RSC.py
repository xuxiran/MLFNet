import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from OURmodels.Base import FeatureCNN,GNet
import torch.autograd as autograd
from torchsummary import summary
import config as cfg

class RSC(nn.Module):
    def __init__(self, device, decision_window):
        super(RSC, self).__init__()
        self.model = rsc(decision_window)
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
            all_f, all_p = self.model(X[6].to(device), X[7].to(device), X[8].to(device), X[9].to(device), X[10].to(device))

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
        print(f"RSC Train Accuracy: {all_correct/count}")

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X,label,identities) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                _, out = self.model(X[6].to(device), X[7].to(device), X[8].to(device), X[9].to(device), X[10].to(device))
                result = np.append(result, torch.max(out, 1)[1].cpu().numpy())
                gt = np.append(gt, label.numpy())
        return result, gt

class rsc(nn.Module):
    def __init__(self, decision_window):
        super(rsc, self).__init__()
        self.feature_cnn = FeatureCNN(decision_window)
        self.g_net = GNet(2, 110)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = x1.unsqueeze(dim=1)
        x2 = x2.unsqueeze(dim=1)
        x3 = x3.unsqueeze(dim=1)
        x4 = x4.unsqueeze(dim=1)
        x5 = x5.unsqueeze(dim=1)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.feature_cnn(x)
        out = self.g_net(x)
        return x,out

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decision_window = 64
    model = rsc(decision_window).to(device)

    x1 = torch.rand((10, decision_window,10,11)).to(device)
    x2 = torch.rand((10, decision_window,10,11)).to(device)
    x3 = torch.rand((10, decision_window,10,11)).to(device)
    x4 = torch.rand((10, decision_window,10,11)).to(device)
    x5 = torch.rand((10, decision_window,10,11)).to(device)

    out = model(x1, x2, x3, x4, x5)
    summary(model, [(decision_window,10,11), (decision_window,10,11), (decision_window,10,11), (decision_window,10,11), (decision_window,10,11)])
