import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, target_size = (64, 64), is_grayscale = False):
        """
        Args:
            root_dir (str): 数据集根目录（按类别分文件夹存放）
            target_size (tuple): 统一调整的图像尺寸（高度, 宽度）
            is_grayscale (bool): 是否转换为灰度图
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.is_grayscale = is_grayscale
        
        # 获取所有子文件夹名称（可以是中文/韩文）
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.classes.sort()  # 按名称排序（确保每次运行顺序一致）
        # print(self.classes)
        
        self.image_paths = []
        self.labels = []
        self.dict = {}
        
        # 遍历文件夹，收集图像路径和标签
        for label, class_name in enumerate(self.classes):
            self.dict[label] = class_name
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(label)  # 标签是整数，与文件夹名称无关

        
        # 定义数据预处理流程
        self.transform = transforms.Compose([
            transforms.Resize(target_size),  # 统一图像尺寸
            transforms.Grayscale(num_output_channels=1) if is_grayscale else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),           # 转为 Tensor [0,1]
            transforms.Normalize((0.5,), (0.5,)) if is_grayscale else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

'''

your_dataset/
    class_0/
        img1.jpg
        img2.jpg
        ...
    class_1/
        img1.jpg
        ...
    ...

'''

class Generator(nn.Module):
    def __init__(self, latent_dim = 100, img_channels = 1, img_size = 64, num_classes = 10):
        super(Generator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 标签嵌入层（假设类别数为 num_classes）
        self.label_embed = nn.Embedding(num_classes, latent_dim)
        
        temp_size = img_size
        while (temp_size % 2 == 0): temp_size //= 2
        if (temp_size == 1): base_size = 4
        else: base_size = temp_size if (temp_size > 4) else temp_size * 2
        
        self.base_size = base_size

        self.fc = nn.Sequential( # 全连接层
            nn.Linear(2 * latent_dim, 256 * base_size * base_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        layers = []
        in_channels = 256
        out_channels = 128
        
        while base_size < img_size:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels
            out_channels //= 2
            base_size *= 2

        layers.append(nn.ConvTranspose2d(in_channels, img_channels, 3, 1, 1))
        layers.append(nn.Tanh())

        self.deconv = nn.Sequential(*layers)

    
    def forward(self, noise, labels):
        # 将噪声和标签拼接
        label_embed = self.label_embed(labels)
        combined = torch.cat((noise, label_embed), dim=1)
        
        out = self.fc(combined)
        
        out = out.view(out.size(0), 256, self.base_size, self.base_size)
        
        img = self.deconv(out)
        return img



class Discriminator(nn.Module):
    def __init__(self, img_channels = 1, img_size = 64, num_classes = 10):
        super(Discriminator, self).__init__()
        self.img_channels = img_channels
        self.img_size = img_size
        
        # 标签嵌入层
        self.label_embed = nn.Embedding(num_classes, img_size * img_size) # 将标签生成一个和图片一样大小的向量
        
        in_channels = img_channels + 1# 初始输入通道为 原始图片通道数 + 标签通道
        out_channels = 64 # 初始化卷积后输出通道
        layers = [] # 卷积层
        
        while (img_size > 4):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)) # 当kernel_size = 4, stride = 2, padding = 1时 image_size //= 2
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.BatchNorm2d(out_channels))
            
            in_channels = out_channels
            out_channels *= 2
            img_size //= 2


        # 卷积层（逐步下采样）
        self.conv = nn.Sequential(*layers)
        
        # 全连接层输出真/假概率
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * img_size * img_size, 1), # 全连接层 in_channels * deal_img_size * deal_img_size  -> 1
            nn.Sigmoid()
        )
    
    
    def forward(self, img, labels):
        # 将标签转换为与图像相同尺寸的通道
        batch_size = img.size(0)
        label_embed = self.label_embed(labels).view(batch_size, 1, self.img_size, self.img_size)
        combined = torch.cat((img, label_embed), dim=1)
        
        out = self.conv(combined)
        validity = self.fc(out)
        return validity



if __name__ == "__main__":
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    '''
    target_size
    is_grayscale
    root_dir
    latent_dim
    batch_size
    num_epochs
    '''

    # 超参数（根据你的数据集调整参数）
    target_size = (96, 96)       # 统一调整后的图像尺寸（高度, 宽度）
    is_grayscale = False         # 是否为灰度图（彩色图设为 False）
    root_dir = "./face_data/"
    latent_dim = 100
    batch_size = 32
    num_epochs = 1


    dataset = CustomImageDataset(root_dir = root_dir, target_size = target_size, is_grayscale = is_grayscale)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    img_channels = 1 if is_grayscale else 3  # 根据是否为灰度图设置通道数
    img_size = target_size[0]                # 图像尺寸（假设为正方形）
    num_classes = len(dataset.classes)       # 类别数（根据数据集自动获取）


    # 初始化模型
    generator = Generator(latent_dim, img_channels, img_size, num_classes).to(device)
    discriminator = Discriminator(img_channels, img_size, num_classes).to(device)

    # 加载参数
    generator.load_state_dict(torch.load("./model/20250308164407/generator.pth"))
    discriminator.load_state_dict(torch.load("./model/20250308164407/discriminator.pth"))

    
    # 定义损失函数和优化器
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    
    # 训练模型
    for epoch in range(num_epochs):
        print(len(dataloader))
        for i, (real_imgs, real_labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)
            
            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()
            
            # 生成假图像
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
            fake_imgs = generator(noise, fake_labels)
            
            # 计算判别器损失
            real_validity = discriminator(real_imgs, real_labels)
            fake_validity = discriminator(fake_imgs.detach(), fake_labels)
            real_loss = adversarial_loss(real_validity, torch.ones(batch_size, 1).to(device))
            fake_loss = adversarial_loss(fake_validity, torch.zeros(batch_size, 1).to(device))
            d_loss = (real_loss + fake_loss) / 2
            
            d_loss.backward()
            optimizer_D.step()
            
            # ---------------------
            #  训练生成器
            # ---------------------
            for _ in range(2):
                optimizer_G.zero_grad()
                noise = torch.randn(batch_size, latent_dim).to(device)
                fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
                fake_imgs = generator(noise, fake_labels)
                validity = discriminator(fake_imgs, fake_labels)
                
                g_loss = adversarial_loss(validity, torch.ones(batch_size, 1).to(device))
                
                g_loss.backward()
                optimizer_G.step()
            
        # 打印训练状态
        print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")


    
    
    
    def generate_image(class_label, num_samples=1):
        noise = torch.randn(num_samples, latent_dim).to(device)
        labels = torch.tensor([class_label] * num_samples).to(device)
        generator.eval()
        with torch.no_grad():
            fake_imgs = generator(noise, labels).cpu()
        # 反标准化到 [0,1]
        fake_imgs = 0.5 * fake_imgs + 0.5
        return fake_imgs

    class_to_generate = 0  # 根据你的类别标签调整
    generated_images = generate_image(class_to_generate, num_samples=5)

    # 可视化
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        ax = axes[i]
        if is_grayscale:
            ax.imshow(generated_images[i].squeeze(), cmap='gray')
        else:
            ax.imshow(generated_images[i].permute(1, 2, 0))  # 调整通道顺序为 HWC
        ax.set_title(f"Class {class_to_generate}")
        ax.axis('off')
    plt.show()
    


'''
关键调整说明
数据集路径：修改 CustomImageDataset 中的 root_dir 为你的数据集路径。

图像尺寸：调整 target_size 为你的目标尺寸（如 (128, 128)）。

通道数：根据是否为灰度图设置 is_grayscale。

类别数：代码会自动从数据集文件夹结构获取类别数。

生成器和判别器结构：根据输入尺寸自动调整网络层参数。

'''