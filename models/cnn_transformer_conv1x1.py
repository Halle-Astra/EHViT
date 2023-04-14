# cnn_transformer.py
# v1.1
import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class CNNTransformer(nn.Module):
    def __init__(self, in_channels, d_model, nhead, num_layers):
        super(CNNTransformer, self).__init__()
        self.cnn_encoder = CNNEncoder(in_channels, d_model)

        self.positional_encoding = nn.Parameter(torch.zeros(1, d_model, 1, 1))

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        self.decoder = nn.Conv2d(d_model, 1, 1)  # 这个也有size问题

        self.d_model = d_model

    def forward(self, x):
        N, C, H, W = x.shape
        x = self.cnn_encoder(x)
        x = x + self.positional_encoding  # Add positional encoding
        # output here is (N,d_model,H,W)
        # x = x.flatten(2).permute(2, 0, 1)  # Reshape and transpose the input for transformer encoder
        # 到@这里flatten+permute@出来@应该是（H*W, N, C）[=(H*W,N,d_model)]. Then the output of original transformer_encoder is (H*W, N, C), too @        x
        x = x.flatten(2).permute(2, 0, 1)
        x = self.transformer_encoder(x)
        # assert N == 1, "Batch size must be 1, for this transformer meanning for calculating space feature correlation between pixels."
        # x = torch.unsqueeze(x, dim=1)
        x = x.view((H, W, N, self.d_model)).permute((2, 3, 0, 1))

        x = self.decoder(x)
        # print("INFO INSPECTING JINGDONG, x.shape after decoder is,", x.shape)
        return x


if __name__ == "__main__":
    from torchsummary import summary

    # 设定超参数
    d_model = 128
    nhead = 1
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 20
    batch_size = 1

    ### important params for mem estimation
    img_h = 187  # 375
    img_w = 624  # 1242

    device = torch.device("cuda")
    model = CNNTransformer(2, d_model, nhead, num_layers).to(device)

    ret = summary(model, input_size=(2, 375, 1242), dtypes=torch.float32)

    # trainable_vars = ret.split('\n')[-2].split(':')[-1].replace(",","")
    # trainable_vars = trainable_vars.strip()
    trainable_vars = ret.trainable_params
    # trainable_vars = eval(trainable_vars)
    trainv_mem = trainable_vars * img_h * img_w * batch_size / 8 / 1000 / 1000 / 1000
    print("The cost of Inference mem will be ", trainv_mem, "Gb")
