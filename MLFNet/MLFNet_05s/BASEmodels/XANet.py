import torch
import torch.nn as nn
import torch.nn.functional as F
import config as cfg
import math
import numpy as np
import tqdm
from torchsummary import summary

class XANet(nn.Module):
    def __init__(self, device, decision_window):
        super(XANet, self).__init__()
        self.model = xanet(decision_window)
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
        print(f"XANet Train Accuracy: {all_correct/count}")

    def test(self, test_loader, device):
        self.model.eval()
        with torch.no_grad():
            result = np.array([])
            gt = np.array([])
            for iter, (X, y, z) in enumerate(tqdm.tqdm(test_loader, position=0, leave=True), start=1):
                convScore = self.model(X[2].to(device))
                result = np.append(result, torch.max(convScore, 1)[1].cpu().numpy())
                gt = np.append(gt, y.numpy())
        return result, gt


# -------------------STAnet---------------------------

def transpose_qkv(X, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Defined in :numref:`sec_multihead-attention`"""
    # Shape of input `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`).
    # Shape of output `X`:
    # (`batch_size`, no. of queries or key-value pairs, `num_heads`,
    # `num_hiddens` / `num_heads`)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # Shape of output `X`:
    # (`batch_size`, `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    X = X.permute(0, 2, 1, 3)

    # Shape of `output`:
    # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
    # `num_hiddens` / `num_heads`)
    return X.reshape(-1, X.shape[2], X.shape[3])


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`subsec_additive-attention`"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # Shape of `queries`: (`batch_size`, no. of queries, `d`)
    # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)

    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(self.attention_weights), values)


def transpose_output(X, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Defined in :numref:`sec_multihead-attention`"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Defined in :numref:`sec_multihead-attention`"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)

        # 很奇怪，但没办法，就是这么定义的
        self.W_q = nn.Linear(query_size, num_hiddens * 10, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens * 10, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        # Shape of `queries`, `keys`, or `values`:
        # (`batch_size`, no. of queries or key-value pairs, `num_hiddens`)

        # After transposing, shape of output `queries`, `keys`, or `values`:
        # (`batch_size` * `num_heads`, no. of queries or key-value pairs,
        # `num_hiddens` / `num_heads`)

        # 在我们的self_attention中，queries,keys,values都是X
        # transpose_qkv我们的目的是为了将X的最后一维分成num_heads份，然后将前两维交换，这样就可以并行计算，不必再去看源码了
        # 第一维会变成batch_size*num_heads，在我们的例子中，就是64*4=256
        # 也就是说，queries、keys、values都是(256,10,8)，第一维都是256，第二维是10，第三维是32/4=8
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # Shape of `output`: (`batch_size` * `num_heads`, no. of queries,
        # `num_hiddens` / `num_heads`)
        # 执行DotProductAttention操作
        # 其实到这里才是自注意力的核心，我们将queries、keys、values都传进去，然后返回一个output
        # 在此之前只是通过将“多头”转移到batch_size这一维度上，这里才开始真正的自注意力
        # 这里的output的维度是(256,10,8)，第一维是256，第二维是10，第三维是32/4=8
        output = self.attention(queries, keys, values)

        # Shape of `output_concat`:
        # (`batch_size`, no. of queries, `num_hiddens`)
        # 这里相当于自注意力的“后处理”，将那些“多头”拼接回来，从而重新变成(64,10,32)
        output_concat = transpose_output(output, self.num_heads)

        # 这里的W_o是一个全连接层，将(64,10,32)变成(64,10,32)
        # 也就是说，这里的W_o是一个恒等映射，所以最终的输出就是(64,10,32)
        # 这里final_output就是我们MultiHeadAttention的最终输出

        return output_concat


class xanet(nn.Module):

    def __init__(self, decision_window):
        super(xanet, self).__init__()

        self.batchnorm = nn.BatchNorm1d(decision_window)
        self.conv2d_left = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv2d_right = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pooling = torch.nn.MaxPool2d(kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))

        self.conv1d_left = nn.Conv1d(in_channels=60, out_channels=5, kernel_size=3, padding=1)
        self.conv1d_right = nn.Conv1d(in_channels=60, out_channels=5, kernel_size=3, padding=1)

        self.linear2 = nn.Linear(8, 64)

        self.dropout = 0.3

        # 时间注意力
        self.attention = MultiHeadAttention(key_size=5, query_size=5,
                                            value_size=5, num_hiddens=5, num_heads=1, dropout=self.dropout)

        # 全连接
        self.fc1 = nn.Linear(decision_window//4*10, 2)

    def forward(self, E):

        E_left = E[:, :, :, :6]
        E_right = E[:, :, :, 5:]
        # We could not get 32 electrodes, because the number of electrodes in one line is odd
        # could not be divided by 2
        E_left = E_left.reshape(E_left.shape[0], E_left.shape[1], -1)
        E_right = E_right.reshape(E_right.shape[0], E_right.shape[1], -1)

        E_left = self.batchnorm(E_left)
        E_right = self.batchnorm(E_right)

        E_left = E_left.unsqueeze(dim=1)
        E_right = E_right.unsqueeze(dim=1)

        E_left = self.conv2d_left(E_left)
        E_right = self.conv2d_right(E_right)

        E_left = self.relu(E_left)
        E_right = self.relu(E_right)

        E_left = self.pooling(E_left)
        E_right = self.pooling(E_right)

        E_left = E_left.squeeze(1)
        E_right = E_right.squeeze(1)

        E_left = E_left.permute(0, 2, 1)
        E_right = E_right.permute(0, 2, 1)

        E_left = self.conv1d_left(E_left)
        E_right = self.conv1d_right(E_right)

        E_left = E_left.permute(0, 2, 1)
        E_right = E_right.permute(0, 2, 1)

        E_left = E_left.unsqueeze(dim=1)
        E_right = E_right.unsqueeze(dim=1)

        E_left = self.pooling(E_left)
        E_right = self.pooling(E_right)

        E_left = E_left.squeeze(1)
        E_right = E_right.squeeze(1)

        E_left_out = self.attention(E_right, E_left, E_left)
        E_right_out = self.attention(E_left, E_right, E_right)

        E_out = torch.cat((E_left_out, E_right_out), 2)
        E_out = E_out.reshape(E_out.shape[0], -1)

        final_out = self.fc1(E_out)

        return final_out

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decision_window = 128
    eeg = torch.randn(8, decision_window, 10, 11).to(device)
    model = xanet(decision_window).to(device)
    out = model(eeg)
    summary(model, (decision_window, 10, 11))
    print("done")