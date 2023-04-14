# train.py
import os
import subprocess
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
# 导入自定义的类
from models.cnn_transformer import CNNTransformer
from utils.dataset import StereoMatchingDataset

from utils.logger import  Logger

logger = Logger("./log.txt")
def print(*args, **kwargs):
    logger(*args)

#####################
# This params group only cost 1300MB GPU Mem (Used the cnn_transformer.py which had been revised by jingdong)
# # 设定超参数
# d_model = 2
# nhead = 1
# num_layers=0
# learning_rate = 1e-4
# num_epochs = 20
# batch_size = 1

#####################
# This params group need 808.10GB of GPU Mem (Used the cnn_transformer.py which had been revised by jingdong)
# # 设定超参数
# d_model = 2
# nhead = 1
# num_layers=1
# learning_rate = 1e-4
# num_epochs = 20
# batch_size = 1
# Analysis: The nn.TransformerEncoderLayer is implemented for NLP, the sequence form.
# So the depth of gradient caculating for Multi-head Attention is the number of pixels (=H*W=375*1242=465.75K times!).
# A part of implementation is attached as following (in C:\Users\halle\miniconda3\envs\torch\Lib\site-packages\torch\nn\functional.py).
# It is implemented recursively, it's a little strange.So it may have a very large number of # gradient depth as I said.
# But I don't find the condition of quit the recursive routine, my opinion is also may be a wrong conclusion.
# I'm not familiar enough with NLP and Transformer.t5j;p
#
# tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
#     if has_torch_function(tens_ops):
#         return handle_torch_function(
#             multi_head_attention_forward,
#             tens_ops,

#####################
# This params group need 2.742GB of GPU Mem (Used the cnn_transformer.py which had been revised by jingdong)
# # 设定超参数
# d_model = 128
# nhead = 1
# num_layers=0
# learning_rate = 1e-4
# num_epochs = 20
# batch_size = 1

# 设定超参数
d_model = 128
nhead = 1
num_layers = 1
learning_rate = 1e-3
num_epochs = 20
batch_size = 1

# 数据集
data_dir = 'data'
dataset = StereoMatchingDataset(data_dir, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# 创建模型、优化器和损失函数
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model = CNNTransformer(2, d_model, nhead, num_layers).to(device)
criterion = nn.MSELoss() # 另一个爆炸的地方
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 2, 3, 7, 10], gamma=0.1)

tensorboard_root = "logs_tensorboard/{}".format(time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())))
TbWriter = SummaryWriter(tensorboard_root)

iter_counter = 0
tb_iters = 40
# 训练过程
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for iter, data in enumerate(dataloader):
        iter += 1 # make it begin from 1
        left_image, right_image, depth = data
        left_image = left_image.to(device)
        right_image = right_image.to(device)
        depth = depth.to(device)

        iter_counter +=1  # for tensorboard

        optimizer.zero_grad()
        output = model(torch.cat((left_image, right_image), dim=1))
        # 这里的输出是不对的，两个维度不一样，应该是会做矩阵广播导致更加爆炸
        # print("="*20+"shape of output\t",output.shape,"="*20)
        # print("="*20+"shape of depth\t", depth.shape, "="*20)
        """
        ====================shape of output	 torch.Size([465750, 1, 1]) ====================
        ====================shape of depth	 torch.Size([1, 1, 375, 1242]) ====================
        """
        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()

        train_loss += loss.detach().item()
        average_loss = train_loss/iter

        print("INFO INSPECTING JINGDONG, Epoch [{}/{}], Iter [{}/{}], loss now is, {}, average loss is {}".format(
                epoch, num_epochs,
                iter, len(dataloader), loss.detach().item(),
                vars().get("average_loss", average_loss)
        ))

        if iter_counter % tb_iters == 0:
            model_params = model.state_dict()
            for param_key in model_params:
                TbWriter.add_histogram('net/{}'.format(param_key), model_params[param_key], iter_counter)
            TbWriter.add_scalar("train/average loss", average_loss, iter_counter)
            TbWriter.add_scalar('train/learning_rate',
                                optimizer.state_dict()['param_groups'][0]['lr'],
                                iter_counter)

            nvidia_info = subprocess.check_output('nvidia-smi')
            nvidia_info = str(nvidia_info, 'utf-8')
            print(nvidia_info)

    scheduler.step()

    # train_loss /= len(dataloader)
    print(f"Epoch: {epoch + 1}, Loss: {train_loss:.6f}")
    model_root = "saving_models"
    if not os.path.exists(model_root):
        os.mkdir(model_root)
    model_path = "MODEL_mean-loss{}_{}.pth".format(average_loss, time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time())))
    optim_path = model_path.replace("MODEL", "OPTIM")
    model_path = os.path.join(model_root, model_path)
    optim_path = os.path.join(model_root, optim_path)
    torch.save(model.state_dict(), model_path)  # 还是不能偷懒直接存整个模型，可能是版本问题，可能暂时只有1.10版本可以
    torch.save(optimizer.state_dict(), optim_path)





