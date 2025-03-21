import shutil, time, os, zipfile, warnings, os, io, base64, torch, json, random
from GAN_model import *
import flet as ft
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
warnings.filterwarnings("ignore")


def main(page: ft.Page):
    # 页面基础设置
    page.title = "GAN模型平台"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0
    page.window_width = 1200
    page.window_height = 800

    # 导航索引状态管理
    selected_index = 0

    parameter = {"device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 "train_generator": None, "train_discriminator": None, "train_generator_parameter": [], "train_discriminator_parameter": [],
                 "generation_generator": None,
                 "train_dataset": None,
                 "train_target_size": None, "train_is_grayscale": None, "train_root_dir": None, "train_latent_dim": None,
                 "train_batch_size": None, "train_data": [], "train_dataset": None,
                 "generation_latent_dim": None, "generation_target_size": None, "generation_is_grayscale": None,
                 "generation_num_classes": None, "generation_model": [], "generation_target": [], "generation_dict": None,
                 }
    print(f"Using device: {parameter['device']}")
#--------------------------------------------------------------------------
    def handle_navigation(e):
        nonlocal selected_index
        selected_index = e.control.selected_index
        update_display()
        page.update()

    # 导航栏设置（使用新枚举规范）
    rail = ft.NavigationRail(
        selected_index=selected_index,
        label_type=ft.NavigationRailLabelType.ALL,
        min_width=100,
        min_extended_width=300,
        group_alignment=-0.9,
        destinations=[ 
            ft.NavigationRailDestination(
                icon=ft.icons.TRAIN,
                selected_icon=ft.icons.TRAIN_OUTLINED,
                label="模型训练"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.AUTO_GRAPH,
                selected_icon=ft.icons.AUTO_GRAPH_OUTLINED,
                label="内容生成"
            ),
            ft.NavigationRailDestination(
                icon=ft.icons.SETTINGS,
                selected_icon=ft.icons.SETTINGS_OUTLINED,
                label="系统设置"
            ),
        ],
        on_change=handle_navigation,
    )

    def grayscale_selected(e, mode):
        selected_value = e.control.value
        if (selected_value == "灰度图"): parameter[mode] = True         # 是否为灰度图（彩色图设为 False）
        if (selected_value == "彩色图"): parameter[mode] = False         # 是否为灰度图（彩色图设为 False）
        # print(selected_value, mode)

        page.update()
    
    def handle_upload(e: ft.FilePickerResultEvent, file_type, text_selected_data, text_upload_status, file_list):
        if e.files:
            # 验证文件类型
            if not all(f.name.endswith('.' + file_type) for f in e.files):
                text_upload_status.value = "错误：只支持" + file_type + "文件"
                text_upload_status.color = ft.colors.RED
                text_upload_status.visible = True
            else:
                # 获取文件信息
                file_info = "\n".join([f.name for f in e.files])
                text_selected_data.value = f"已选择文件：\n{file_info}"
                text_upload_status.value = "文件验证通过！"
                text_upload_status.color = ft.colors.BLUE
                text_upload_status.visible = True
                
                # 存储上传的文件对象
                parameter[file_list].clear()
                parameter[file_list].extend(e.files)

            page.update()
    
#--------------------------------------------------------------------------
    # 添加文件选择器控件
    data_picker = ft.FilePicker()
    generator_picker = ft.FilePicker()
    discriminator_picker = ft.FilePicker()
    page.overlay.append(data_picker)
    page.overlay.append(generator_picker)
    page.overlay.append(discriminator_picker)
    page.update()
    
    # 存储上传状态
    data_upload_status = ft.Text(visible=False, color=ft.colors.GREEN)
    selected_data = ft.Text()
    generator_upload_status = ft.Text(visible=False, color=ft.colors.GREEN)
    selected_generator = ft.Text()
    discriminator_upload_status = ft.Text(visible=False, color=ft.colors.GREEN)
    selected_discriminator = ft.Text()

    train_length_input = ft.TextField(label = "训练图片长度", width = 100, keyboard_type = ft.KeyboardType.NUMBER)
    train_width_input = ft.TextField(label = "训练图片宽度", width = 100, keyboard_type = ft.KeyboardType.NUMBER)

    

    # 创建单选框组
    train_grayscale_group = ft.RadioGroup(
        content=ft.Row(
            [
                ft.Radio(value = "灰度图", label = "灰度图"),
                ft.Radio(value = "彩色图", label = "彩色图"),
            ]
        ),
        on_change = lambda e: grayscale_selected(e, "train_is_grayscale")
    )


    # 文件选择回调
    data_picker.on_result = lambda e: handle_upload(e, "zip", selected_data, data_upload_status, "train_data")
    generator_picker.on_result = lambda e: handle_upload(e, "pth", selected_generator, generator_upload_status, "train_generator_parameter")
    discriminator_picker.on_result = lambda e: handle_upload(e, "pth", selected_discriminator, discriminator_upload_status, "train_discriminator_parameter")

    # 用于模拟训练的时间和进度
    epoch_input = ft.TextField(label="训练Epoch数", value="10", width = 150, keyboard_type=ft.KeyboardType.NUMBER)
    progress_bar = ft.ProgressBar(value = 0, width = 300)
    progress_text = ft.Text("0%", size = 16)
    estimated_time_text = ft.Text("预计时间: 0秒", size=16)

    
    def update_progress(total_count, current_count, single_time):
        progress = current_count / total_count
        progress_bar.value = progress
        # 更新进度百分比
        progress_text.value = f"{int(progress * 100)}%"
        # 预计时间
        estimated_time = int((total_count - current_count) * single_time) + 1
        estimated_time_text.value = f"预计时间: {estimated_time}秒"
        page.update()


    def start_training(_):
        try:
            if not parameter["train_data"]:
                raise ValueError("请先上传数据集")
            if (train_length_input.value == "" or train_width_input.value == ""):
                raise ValueError("请先输入训练图片大小")
            if (int(train_length_input.value) <= 0 or int(train_width_input.value) <= 0):
                raise ValueError("训练图片大小必须大于0")
            if (parameter["train_is_grayscale"] == None):
                raise ValueError("请先选择色度值")
            if (epoch_input.value == ""):
                raise ValueError("请输入Epoch数")
            if (int(epoch_input.value) <= 0):
                raise ValueError("Epoch数必须大于0")

            # 禁用按钮
            start_button.disabled = True
            export_button.disabled = True
            progress_text.value = "开始处理数据"
            page.update()

            
            for file in parameter["train_data"]:
                # 获取上传文件的路径
                file_path = file.path
                # 将文件移动到当前工作目录
                target_path = os.path.join(os.getcwd(), file.name)
                shutil.copy(file_path, target_path)

                try:
                    # 打开压缩包
                    with zipfile.ZipFile(file.path, 'r') as zip_ref:
                        # 获取压缩包内所有文件和文件夹的名字
                        names = zip_ref.namelist()
                        parameter["train_root_dir"] = "./" + names[0]
                except Exception as ex:
                    print(f"处理压缩包 {file.name} 时出错：{ex}")

                # 解压文件
                print("开始解压文件")
                try:
                    # 解压文件到当前工作目录
                    with zipfile.ZipFile(target_path, 'r') as zip_ref:
                        zip_ref.extractall(os.getcwd())
                    data_upload_status.value = f"文件 {file.name} 解压成功！"
                except zipfile.BadZipFile:
                    data_upload_status.value = "错误：ZIP文件解压失败"
                    data_upload_status.color = ft.colors.RED
                    data_upload_status.visible = True
                    page.update()
                    return
                finally:
                    # 删除已移动的ZIP文件
                    os.remove(target_path)
                print(f"文件已解压到: {os.getcwd()}")
            
            # 开始训练
            progress_bar.value = 0  # 重置进度条
            progress_text.value = "0%"
            estimated_time_text.value = "预计时间: 0秒"
            

            # 超参数（根据你的数据集调整参数）
            parameter["train_target_size"] = (int(train_length_input.value), int(train_width_input.value))       # 统一调整后的图像尺寸（高度, 宽度）
            
            parameter["train_latent_dim"] = 100
            parameter["train_batch_size"] = 32
            num_epochs = int(epoch_input.value)

            parameter["train_dataset"] = CustomImageDataset(root_dir = parameter["train_root_dir"], target_size = parameter["train_target_size"], is_grayscale = parameter["train_is_grayscale"])
            print(parameter["train_dataset"].dict)

            dataloader = DataLoader(parameter["train_dataset"], batch_size = parameter["train_batch_size"], shuffle = True)
            img_channels = 1 if parameter["train_is_grayscale"] else 3  # 根据是否为灰度图设置通道数
            img_size = parameter["train_target_size"][0]                # 图像尺寸（假设为正方形）
            num_classes = len(parameter["train_dataset"].classes)       # 类别数（根据数据集自动获取）

            # 初始化模型
            parameter["train_generator"] = Generator(parameter["train_latent_dim"], img_channels, img_size, num_classes).to(parameter["device"])
            parameter["train_discriminator"] = Discriminator(img_channels, img_size, num_classes).to(parameter["device"])

            # 加载模型参数
            if parameter["train_generator_parameter"]:
                for model in parameter["train_generator_parameter"]:
                    parameter["train_generator"].load_state_dict(torch.load(model.path))

            if parameter["train_discriminator_parameter"]:
                for model in parameter["train_discriminator_parameter"]:
                    parameter["train_discriminator"].load_state_dict(torch.load(model.path))

            # 定义损失函数和优化器
            adversarial_loss = nn.BCELoss()
            optimizer_G = optim.Adam(parameter["train_generator"].parameters(), lr=0.0002, betas=(0.5, 0.999))
            optimizer_D = optim.Adam(parameter["train_discriminator"].parameters(), lr=0.0001, betas=(0.5, 0.999))

            # 训练模型
            print("开始训练")
            data_count = len(dataloader)
            for epoch in range(num_epochs):
                for i, (real_imgs, real_labels) in enumerate(dataloader):
                    if (i % 20 == 0): time_begin = time.time()
                    parameter["train_batch_size"] = real_imgs.size(0)
                    real_imgs = real_imgs.to(parameter["device"])
                    real_labels = real_labels.to(parameter["device"])
                    real_imgs_noisy = real_imgs + 0.1 * torch.randn_like(real_imgs)  # 添加高斯噪声

                    optimizer_D.zero_grad()
                    
                    # 生成假图像
                    noise = torch.randn(parameter["train_batch_size"], parameter["train_latent_dim"]).to(parameter["device"])
                    fake_labels = torch.randint(0, num_classes, (parameter["train_batch_size"],)).to(parameter["device"])
                    fake_imgs = parameter["train_generator"](noise, fake_labels)
                    
                    # 计算判别器损失
                    real_validity = parameter["train_discriminator"](real_imgs_noisy, real_labels)
                    fake_validity = parameter["train_discriminator"](fake_imgs.detach(), fake_labels)
                    real_loss = adversarial_loss(real_validity, torch.full((parameter["train_batch_size"], 1), 0.9).to(parameter["device"]))
                    fake_loss = adversarial_loss(fake_validity, torch.zeros(parameter["train_batch_size"], 1).to(parameter["device"]))
                    d_loss = (real_loss + fake_loss) / 2
                    
                    d_loss.backward()
                    optimizer_D.step()


                    for _ in range(random.randint(3, 6)): # 多次训练生成器
                        optimizer_G.zero_grad()
                        
                        # 每次创建新的噪声 生成新的假图像
                        noise = torch.randn(parameter["train_batch_size"], parameter["train_latent_dim"]).to(parameter["device"])
                        fake_labels = torch.randint(0, num_classes, (parameter["train_batch_size"],)).to(parameter["device"])
                        fake_imgs = parameter["train_generator"](noise, fake_labels)

                        validity = parameter["train_discriminator"](fake_imgs, fake_labels)
                        g_loss = adversarial_loss(validity, torch.ones(parameter["train_batch_size"], 1).to(parameter["device"]))
                        
                        g_loss.backward()
                        optimizer_G.step()

                    if (i % 20 == 0): time_end = time.time()

                    update_progress((num_epochs * data_count), (epoch * data_count) + i, time_end - time_begin)
                    
                # 打印训练状态
                print(f"Epoch [{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
            # 训练结束后恢复按钮可点击状态
            update_progress((num_epochs * data_count), (num_epochs * data_count) + 1, 1)
            start_button.disabled = False
            export_button.disabled = False
            page.update()

        except ValueError as e:
            progress_text.value = f"错误: {e}"
            page.update()


    def export_parameter(_):
        nowtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        new_folder = ''.join([x for x in nowtime if x.isdigit()])
        
        os.makedirs(os.getcwd() + "/model/" + new_folder)

        if (os.path.exists(f"./model/{new_folder}")):
            torch.save(parameter["train_generator"].state_dict(), f"./model/{new_folder}/generator.pth")
            torch.save(parameter["train_discriminator"].state_dict(), f"./model/{new_folder}/discriminator.pth")
            with open(f"./model/{new_folder}/dict.json", "w", encoding = "utf-8") as f:
                json.dump(parameter["train_dataset"].dict, f, ensure_ascii = False, indent = 4)
            progress_text.value = f"模型已保存在./model/{new_folder}文件夹"
            page.update()
            print("模型参数已保存！")
        else:
            progress_text.value = f"模型保存失败"
            page.update()
            print("模型参数保存失败！")

    start_button = ft.ElevatedButton("开始训练", 
        icon=ft.icons.PLAY_ARROW,
        on_click=start_training
    )

    export_button = ft.ElevatedButton("导出模型", 
        icon = ft.icons.PLAY_ARROW,
        on_click = export_parameter,
        disabled = True
    )
    
    def training_view():
        return ft.Container(
            content=ft.Column([ 
                ft.Text("模型训练", size=24, weight=ft.FontWeight.BOLD),
                ft.Divider(),

                ft.Row([
                    # 上传文件区域
                    ft.Row([
                        ft.ElevatedButton(
                            "上传数据集(ZIP)",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: data_picker.pick_files(
                                allowed_extensions=["zip"],
                                allow_multiple=False
                            )
                        ),
                        ft.Column([
                            selected_data,
                            data_upload_status
                        ], spacing = 5),
                    ], spacing=10),

                    # 上传生成器模型
                    ft.Row([
                        ft.ElevatedButton(
                            "上传生成器模型(PTH)",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: generator_picker.pick_files(
                                allowed_extensions=["pth"],
                                allow_multiple=True
                            )
                        ),
                        ft.Column([
                            selected_generator,
                            generator_upload_status
                        ], spacing=5),
                    ], spacing=10),

                    # 上传判别器模型
                    ft.Row([
                        ft.ElevatedButton(
                            "上传判别器模型(PTH)",
                            icon=ft.icons.UPLOAD_FILE,
                            on_click=lambda _: discriminator_picker.pick_files(
                                allowed_extensions=["pth"],
                                allow_multiple=True
                            )
                        ),
                        ft.Column([
                            selected_discriminator,
                            discriminator_upload_status
                        ], spacing=5),
                    ], spacing=10),
                ]),
                

                ft.Row(
                controls=[
                    train_length_input,
                    train_width_input,
                ],
                spacing=20  # 按钮之间的间距
                ),

                train_grayscale_group,
                # 输入训练Epoch数
                epoch_input,

                # 训练进度条和信息
                ft.Row(
                controls=[
                    progress_bar,
                    progress_text,
                    estimated_time_text,
                ],
                spacing=20  # 按钮之间的间距
                ),
                
                ft.Row(
                controls=[
                    start_button,  # 将按钮添加到视图
                    export_button,  # 将按钮添加到视图
                ],
                spacing=20  # 按钮之间的间距
                )
                
            ],
            spacing=20,
            scroll = ft.ScrollMode.AUTO, # <--- 关键添加项：启用自动滚动
            expand = True,
            ),

            padding=30,
            expand = True,
            alignment=ft.alignment.top_center
        )

#--------------------------------------------------------------------------


    def handle_target_upload(e: ft.FilePickerResultEvent, file_type, text_selected_data, text_upload_status, text_dropdown, file_list):
        if e.files:
            # 验证文件类型
            if not all(f.name.endswith('.' + file_type) for f in e.files):
                text_upload_status.value = "错误：只支持" + file_type + "文件"
                text_upload_status.color = ft.colors.RED
                text_upload_status.visible = True
            else:
                # 获取文件信息
                file_info = "\n".join([f.name for f in e.files])
                text_selected_data.value = f"已选择文件：\n{file_info}"
                text_upload_status.value = "文件验证通过！"
                text_upload_status.color = ft.colors.BLUE
                text_upload_status.visible = True
                
                # 存储上传的文件对象
                parameter[file_list].clear()
                parameter[file_list].extend(e.files)

                # 读取第一个上传的 JSON 文件
                file_path = e.files[0].path  # 获取文件路径
                try:
                    with open(file_path, 'r') as f:
                        parameter["generation_dict"] = json.load(f)
                        text_upload_status.value = "JSON 文件读取成功！"
                except Exception as ex:
                    text_upload_status.value = f"错误：无法读取文件 ({ex})"
                    text_upload_status.color = ft.colors.RED
                    text_upload_status.visible = True
                target_options = [ft.dropdown.Option(f"{parameter['generation_dict'][key]}") for key in parameter["generation_dict"]]

                text_dropdown.options = target_options
                # text_dropdown.value = "0"

            page.update()

    # 创建单选框组
    generation_grayscale_group = ft.RadioGroup(
        content=ft.Row(
            [
                ft.Radio(value = "灰度图", label = "灰度图"),
                ft.Radio(value = "彩色图", label = "彩色图"),
            ]
        ),
        on_change = lambda e: grayscale_selected(e, "generation_is_grayscale")
    )

    generate_text = ft.Text("", size = 16, color = ft.colors.RED)
    generation_length_input = ft.TextField(label = "生成图片长度", width = 100, keyboard_type = ft.KeyboardType.NUMBER)
    generation_width_input = ft.TextField(label = "生成图片宽度", width = 100, keyboard_type = ft.KeyboardType.NUMBER)

    generation_dropdown = ft.Dropdown(width = 200)

    # 添加文件选择器控件
    parameter_picker = ft.FilePicker()
    target_picker = ft.FilePicker()
    page.overlay.append(parameter_picker)
    page.overlay.append(target_picker)
    page.update()
    
    parameter_upload_status = ft.Text(visible = False, color = ft.colors.BLUE)
    selected_parameter = ft.Text()
    
    target_upload_status = ft.Text(visible = False, color = ft.colors.BLUE)
    selected_target = ft.Text()

    parameter_picker.on_result = lambda e: handle_upload(e, "pth", selected_parameter, parameter_upload_status, "generation_model")
    target_picker.on_result = lambda e: handle_target_upload(e, "json", selected_target, target_upload_status, generation_dropdown, "generation_target")

    

    # 生成的图片 UI 组件
    image_display = ft.Image(visible = False)

    def generate_image(class_label, num_samples = 1):
        try:
            print(class_label)
            if (not parameter["generation_model"]):
                raise ValueError("请先上传模型参数")
            if (generation_length_input.value == "" or generation_width_input.value == ""):
                raise ValueError("请先输入生成图片大小")
            if (int(generation_length_input.value) <= 0 or int(generation_width_input.value) <= 0):
                raise ValueError("生成图片大小必须大于0")
            if (parameter["generation_is_grayscale"] == None):
                raise ValueError("请先选择色度值")
            if (not parameter["generation_target"]):
                raise ValueError("请先上传映射字典")
            if (class_label == None):
                raise ValueError("请选择生成目标")
            

            parameter["generation_target_size"] = (int(generation_length_input.value), int(generation_width_input.value))       # 统一调整后的图像尺寸（高度, 宽度）
            parameter["generation_num_classes"] = len(parameter["generation_dict"])

            selected_classes = None
            for key in parameter["generation_dict"]:
                if (parameter["generation_dict"][key] == class_label):
                    selected_classes = int(key)
                    break

            for model in parameter["generation_model"]:
                parameter["generation_latent_dim"] = 100
                img_channels = 1 if parameter["generation_is_grayscale"] else 3  # 根据是否为灰度图设置通道数
                img_size = parameter["generation_target_size"][0]                # 图像尺寸（假设为正方形）

                parameter["generation_generator"] = Generator(parameter["generation_latent_dim"], img_channels, img_size, parameter["generation_num_classes"]).to(parameter["device"])
                parameter["generation_generator"].load_state_dict(torch.load(model.path))

            generate_text.value = ""


            noise = torch.randn(num_samples, parameter["generation_latent_dim"]).to(parameter["device"])
            labels = torch.tensor([selected_classes] * num_samples).to(parameter["device"])
            parameter["generation_generator"].eval()
            with torch.no_grad():
                fake_imgs = parameter["generation_generator"](noise, labels).cpu()
            # 反标准化到 [0,1]
            fake_imgs = 0.5 * fake_imgs + 0.5

            '''
            fig, ax = plt.subplots(1, 1, figsize=(15, 3))
            if parameter["generation_is_grayscale"]:
                ax.imshow(fake_imgs.squeeze(), cmap='gray')
            else:
                ax.imshow(fake_imgs.permute(1, 2, 0))  # 调整通道顺序为 HWC
            ax.set_title(f"Class {class_label}")
            ax.axis('off')
            plt.show()
            '''
            ''''''
            # **调整形状**
            fake_imgs = fake_imgs.squeeze(0)  # **移除 batch 维度，形状变为 (1, 28, 28)**
            
            if fake_imgs.shape[0] == 1:
                fake_imgs = fake_imgs.squeeze(0)  # **如果是灰度图 (1, H, W)，去掉通道维度，变成 (H, W)**
            else:
                fake_imgs = fake_imgs.permute(1, 2, 0)  # **如果是 RGB (3, H, W)，调整成 (H, W, 3)**

            # **转换为 NumPy**
            fake_imgs = (fake_imgs.numpy() * 255).astype(np.uint8)  # **转换为 uint8**

            # **转换为 PIL 图像**
            img = Image.fromarray(fake_imgs)  # **PIL 现在可以正常处理**

            # **放大图片**
            upscale_factor = 1
            original_size = img.size  # (W, H)
            new_size = (original_size[0] * upscale_factor, original_size[1] * upscale_factor)
            img = img.resize(new_size, Image.NEAREST)  # 使用最近邻插值放大图像

            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            img_base64 = base64.b64encode(img_bytes.read()).decode()
            image_display.src_base64 = img_base64
            image_display.visible = True

            page.update()

        except ValueError as e:
            generate_text.color = ft.colors.RED
            generate_text.value = f"错误: {e}"
            page.update()
    

    def download_image():
        try:
            if (not os.path.exists(os.getcwd() + "/generated_image")):
                os.makedirs(os.getcwd() + "/generated_image")

            nowtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            output_path = ''.join([x for x in nowtime if x.isdigit()])
            output_path = os.getcwd() + "/generated_image/" + output_path + ".png"

            img_base64 = image_display.src_base64
            # 将 Base64 解码为二进制数据
            img_data = base64.b64decode(img_base64)
            # 保存为文件
            with open(output_path, "wb") as f:
                f.write(img_data)

            generate_text.color = ft.colors.BLACK
            generate_text.value = f"图片已保存至 {output_path}"
            page.update()
            print(f"图片已保存至 {output_path}")

        except:
            generate_text.value = f"图片保存失败"
            page.update()
            print("图片保存失败！")

    def generate_view():
        # 定义训练视图
        return ft.Container(
            content=ft.Column([ 
                ft.Text("内容生成", size = 24, weight = ft.FontWeight.BOLD),
                ft.Divider(),

                ft.Row([
                    ft.ElevatedButton(
                        "上传模型(PTH)",
                        icon=ft.icons.UPLOAD_FILE,
                        on_click = lambda _: parameter_picker.pick_files(
                            allowed_extensions = ["pth"],
                            allow_multiple = False
                        )
                    ),
                    ft.Column([
                        selected_parameter,
                        parameter_upload_status
                    ], spacing = 5),
                ], spacing = 10),

                ft.Row(
                    controls=[
                        generation_length_input,
                        generation_width_input,
                    ],
                    spacing=20  # 按钮之间的间距
                ),
                generation_grayscale_group,

                ft.Row([
                    ft.ElevatedButton(
                        "上传映射字典(JSON)",
                        icon=ft.icons.UPLOAD_FILE,
                        on_click = lambda _: target_picker.pick_files(
                            allowed_extensions = ["json"],
                            allow_multiple = False
                        )
                    ),
                    ft.Column([
                        selected_target,
                        target_upload_status
                    ], spacing = 5),
                ], spacing = 10),


                ft.Text("选择生成目标", size = 16, weight = ft.FontWeight.BOLD),
                
                generation_dropdown,
                
                ft.Row([
                    ft.ElevatedButton("生成内容",
                                  icon = ft.icons.AUTO_GRAPH,
                                  on_click = lambda e: generate_image(generation_dropdown.value, 1)
                                  ),
                    ft.ElevatedButton("保存内容",
                                  icon = ft.icons.AUTO_GRAPH,
                                  on_click = lambda e: download_image()
                                  ),
                    generate_text,
                ]),
                
                image_display
            ],
            spacing = 20,
            scroll = ft.ScrollMode.AUTO, # <--- 关键添加项：启用自动滚动
            expand = True,
            ),
            padding = 30,
            expand = True,
            alignment=ft.alignment.top_center
        )

#--------------------------------------------------------------------------

    def settings_view():
        return ft.Container(
            content = ft.Column([ 
                ft.Text("系统设置", size=24, weight = ft.FontWeight.BOLD),
                ft.Divider(),
                ft.ElevatedButton("保存设置", icon = ft.icons.SAVE)
            ], spacing = 20),
            padding = 30
        )

    content_area = ft.Container()


#--------------------------------------------------------------------------

    def update_display():
        views = [training_view, generate_view, settings_view]
        content_area.content = views[selected_index]()
        page.update()

    # 构建主界面
    page.add(
        ft.Row([ 
            ft.Container(
                content=rail,
                padding=ft.padding.only(top=20),
                width=200,
                bgcolor=ft.colors.BLUE_GREY if page.theme_mode == ft.ThemeMode.LIGHT else ft.colors.BLUE_GREY_900,
            ),
            ft.Container(
                content=content_area,
                expand=True,
                border_radius=ft.border_radius.all(10),
                shadow=ft.BoxShadow(
                    spread_radius=1,
                    blur_radius=15,
                    color=ft.colors.BLUE_GREY_300,
                    offset=ft.Offset(0, 0),
                ),
                margin=ft.margin.all(20),
                padding=ft.padding.all(10),
            )
        ], expand=True)
    )

    update_display()

ft.app(target = main)