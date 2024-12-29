# 这个 Python 3 环境附带了许多有用的分析库
# 它是由 kaggle/ Python docker 映像定义的：https://github.com/kaggle/docker-python

import numpy as np  # 用于线性代数运算
import pandas as pd  # 用于数据处理，如 CSV 文件的 I/O 操作（如 pd.read_csv）
from matplotlib import pyplot as plt  # 用于数据可视化
import cv2  # 用于图像处理
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # 用于加载和转换图像
from tensorflow.keras.utils import to_categorical, Sequence  # 用于将数据转换为分类形式，以及创建自定义数据生成器类
from tensorflow.keras import backend as K  # 用于使用 Keras 的后端功能
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout  # 用于构建神经网络的各种层
from tensorflow.keras.optimizers import Adadelta, Nadam, Adam  # 用于优化神经网络的优化器
from tensorflow.keras.models import Model, load_model  # 用于创建和加载模型
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard  # 用于训练过程中的回调函数


# 输入数据文件可在“../输入/”目录。
# 例如，运行 this（通过点击 run 或按 Shift+Enter）将列出输入目录中的文件

import os
print(os.listdir("D:/Camvid_Segmentation_Unet/CamVid"))  # 打印指定目录下的文件列表
from glob import glob  # 用于查找匹配文件路径的模式
from pathlib import Path  # 用于处理文件路径
import shutil  # 用于文件操作，如复制、移动等
from tqdm import tqdm_notebook  # 用于显示进度条
from random import sample, choice  # 用于随机采样和选择


# 写入当前目录的任何结果都保存为输出。


dataset_path = Path("D:/Camvid_Segmentation_Unet/CamVid")  # 数据集所在的路径
list(dataset_path.iterdir())  # 获取路径下的文件列表


def tree(directory):
    # 以树状结构打印目录下的文件和子目录
    print(f'+ {directory}')
    for path in sorted(directory.rglob('*')):
        depth = len(path.relative_to(directory).parts)
        spacer = '    ' * depth
        print(f'{spacer}+ {path.name}')


# tree(dataset_path)


train_imgs = list((dataset_path / "train").glob("*.png"))  # 获取训练图像的文件列表
train_labels = list((dataset_path / "train_labels").glob("*.png"))  # 获取训练标签的文件列表
val_imgs = list((dataset_path / "val").glob("*.png"))  # 获取验证图像的文件列表
val_labels = list((dataset_path / "val_labels").glob("*.png"))  # 获取验证标签的文件列表
test_imgs = list((dataset_path / "test").glob("*.png"))  # 获取测试图像的文件列表
test_labels = list((dataset_path / "test_labels").glob("*.png"))  # 获取测试标签的文件列表


# 检查图像和标签的数量是否匹配
(len(train_imgs), len(train_labels)), (len(val_imgs), len(val_labels)), (len(test_imgs), len(test_labels))


img_size = 512


# 断言图像和标签的数量是否相等，若不相等则输出错误信息
assert len(train_imgs) == len(train_labels), "No of Train images and label mismatch"
assert len(val_imgs) == len(val_labels), "No of Train images and label mismatch"
assert len(test_imgs) == len(test_labels), "No of Train images and label mismatch"


# 对图像和标签列表进行排序
sorted(train_imgs), sorted(train_labels), sorted(val_imgs), sorted(val_labels), sorted(test_imgs), sorted(test_labels)


# 检查图像和标签是否一一对应，若不对应则输出错误信息
for im in train_imgs:
    assert dataset_path / "train_labels" / (im.stem + "_L.png") in train_labels, "{im} not there in label folder"
for im in val_imgs:
    assert dataset_path / "val_labels" / (im.stem + "_L.png") in val_labels, "{im} not there in label folder"
for im in test_imgs:
    assert dataset_path / "test_labels" / (im.stem + "_L.png") in test_labels, "{im} not there in label folder"


def make_pair(img, label, dataset):
    # 创建图像和标签的配对列表
    pairs = []
    for im in img:
        pairs.append((im, dataset / label / (im.stem + "_L.png")))
    return pairs


# 生成训练、验证和测试数据的图像-标签对
train_pair = make_pair(train_imgs, "train_labels", dataset_path)
val_pair = make_pair(val_imgs, "val_labels", dataset_path)
test_pair = make_pair(test_imgs, "test_labels", dataset_path)


temp = choice(train_pair)  # 从训练对中随机选择一个
img = img_to_array(load_img(temp[0], target_size=(img_size, img_size)))  # 加载图像并转换为数组
mask = img_to_array(load_img(temp[1], target_size=(img_size, img_size)))  # 加载标签并转换为数组
plt.figure(figsize=(10, 10))  # 创建一个新的图像显示区域
plt.subplot(121)  # 显示图像
plt.imshow(img / 255)
plt.subplot(122)  # 显示标签
plt.imshow(mask / 255)


class_map_df = pd.read_csv(dataset_path / "class_dict.csv")  # 读取类别映射的 CSV 文件


class_map = []  # 存储类别映射信息
for index, item in class_map_df.iterrows():
    class_map.append(np.array([item['r'], item['g'], item['b']]))


len(class_map)


def assert_map_range(mask, class_map):
    # 检查标签中像素值是否在类别映射中
    mask = mask.astype("uint8")
    for j in range(img_size):
        for k in range(img_size):
            assert mask[j][k] in class_map, tuple(mask[j][k])


def form_2D_label(mask, class_map):
    # 将 RGB 标签转换为 2D 标签矩阵
    mask = mask.astype("uint8")
    label = np.zeros(mask.shape[:2], dtype=np.uint8)
    for i, rgb in enumerate(class_map):
        label[(mask == rgb).all(axis=2)] = i
    return label


lab = form_2D_label(mask, class_map)
np.unique(lab, return_counts=True)


class DataGenerator(Sequence):
    # 自定义的数据生成器类，用于为 Keras 生成数据
    def __init__(self, pair, class_map, batch_size=16, dim=(224, 224, 3), shuffle=True):
        # 初始化数据生成器的参数
        self.dim = dim
        self.pair = pair
        self.class_map = class_map
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()


    def __len__(self):
        # 计算每个 epoch 的批次数
        return int(np.floor(len(self.pair) / self.batch_size))


    def __getitem__(self, index):
        # 获取一个批次的数据
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [k for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y


    def on_epoch_end(self):
        # 在每个 epoch 结束时重新洗牌索引
        self.indexes = np.arange(len(self.pair))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, list_IDs_temp):
        # 生成一个批次的数据
        batch_imgs = list()
        batch_labels = list()
        for i in list_IDs_temp:
            # 加载图像，将其转换为数组并归一化
            img = load_img(self.pair[i][0], target_size=self.dim)
            img = img_to_array(img) / 255.
            batch_imgs.append(img)
            # 加载标签，将其转换为数组并转换为分类形式
            label = load_img(self.pair[i][1], target_size=self.dim)
            label = img_to_array(label)
            label = form_2D_label(label, self.class_map)
            label = to_categorical(label, num_classes=32)
            batch_labels.append(label)
        return np.array(batch_imgs), np.array(batch_labels)


# 创建训练、验证数据生成器，并计算步骤数
train_generator = DataGenerator(train_pair + test_pair, class_map, batch_size=4, dim=(img_size, img_size, 3), shuffle=True)
train_steps = train_generator.__len__()
train_steps


X, y = train_generator.__getitem__(1)
y.shape


val_generator = DataGenerator(val_pair, class_map, batch_size=4, dim=(img_size, img_size, 3), shuffle=True)
val_steps = val_generator.__len__()
val_steps


def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    # 定义卷积块，包含两个卷积层和批归一化、激活操作
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    # 定义反卷积块，包含反卷积操作和拼接操作
    y = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    y = concatenate([y, residual], axis=3)
    y = conv_block(y, nfilters)
    return y


def Unet(h, w, filters):
    # 定义 U-Net 模型
    input_layer = Input(shape=(h, w, 3), name='image_input')  # 输入层
    conv1 = conv_block(input_layer, nfilters=filters)  # 第一个卷积块
    conv1_out = MaxPooling2D(pool_size=(2, 2))(conv1)  # 最大池化
    conv2 = conv_block(conv1_out, nfilters=filters * 2)  # 第二个卷积块
    conv2_out = MaxPooling2D(pool_size=(2, 2))(conv2)  # 最大池化
    conv3 = conv_block(conv2_out, nfilters=filters * 4)  # 第三个卷积块
    conv3_out = MaxPooling2D(pool_size=(2, 2))(conv3)  # 最大池化
    conv4 = conv_block(conv3_out, nfilters=filters * 8)  # 第四个卷积块
    conv4_out = MaxPooling2D(pool_size=(2, 2))(conv4)  # 最大池化
    conv4_out = Dropout(0.5)(conv4_out)  # 随机失活
    conv5 = conv_block(conv4_out, nfilters=filters * 16)  # 第五个卷积块
    conv5 = Dropout(0.5)(conv5)  # 随机失活
    # 上采样部分
    deconv6 = deconv_block(conv5, residual=conv4, nfilters=filters * 8)  # 第一个反卷积块
    deconv6 = Dropout(0.5)(deconv6)  # 随机失活
    deconv7 = deconv_block(deconv6, residual=conv3, nfilters=filters * 4)  # 第二个反卷积块
    deconv7 = Dropout(0.5)(deconv7)  # 随机失活
    deconv8 = deconv_block(deconv7, residual=conv2, nfilters=filters * 2)  # 第三个反卷积块
    deconv9 = deconv_block(deconv8, residual=conv1, nfilters=filters)  # 第四个反卷积块
    output_layer = Conv2D(filters=32, kernel_size=(1, 1), activation='softmax')(deconv9)  # 输出层
    model = Model(inputs=input_layer, outputs=output_layer, name='Unet')  # 创建模型
    return model


model = Unet(img_size, img_size, 64)  # 创建 U-Net 模型实例
model.summary()  # 打印模型的摘要信息


# 编译模型，使用 Adam 优化器，交叉熵损失函数和准确率作为评估指标
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mc = ModelCheckpoint(mode='max', filepath='top-weights.weights.h5', monitor='val_acc', save_best_only='True', save_weights_only='True', verbose=1)  # 模型检查点，保存最佳权重
es = EarlyStopping(mode='max', monitor='val_acc', patience=10, verbose=0)  # 早停策略
tb = TensorBoard(log_dir="logs/", histogram_freq=0, write_graph=True, write_images=False)  # TensorBoard 回调
rl = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=5, verbose=1, mode="max", min_lr=0.0001)  # 学习率调整策略
cv = CSVLogger("logs/log.csv", append=True, separator=',')  # 日志记录器
results = model.fit(train_generator, steps_per_epoch=train_steps, epochs=1,  # 训练模型
                  validation_data=val_generator, validation_steps=val_steps, callbacks=[mc, es, tb, rl, cv])


img_mask = choice(val_pair)  # 从验证对中随机选择一个
img = img_to_array(load_img(img_mask[0], target_size=(img_size, img_size)))  # 加载验证图像并转换为数组
gt_img = img_to_array(load_img(img_mask[1], target_size=(img_size, img_size)))  # 加载验证标签并转换为数组


def make_prediction(model, img_path, shape):
    # 对图像进行预测
    img = img_to_array(load_img(img_path, target_size=shape)) / 255.
    img = np.expand_dims(img, axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0], axis=2)
    return labels


pred_label = make_prediction(model, img_mask[0], (img_size, img_size, 3))  # 进行预测
pred_label.shape


def form_colormap(prediction, mapping):
    # 将预测结果转换为彩色图像
    h, w = prediction.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label


pred_colored = form_colormap(pred_label, np.array(class_map))  # 将预测结果转换为彩色图像


plt.figure(figsize=(15, 15))  # 创建一个新的图像显示区域
plt.subplot(131)  # 显示原始图像
plt.title('Original Image')
plt.imshow(img / 255.)
plt.subplot(132)  # 显示真实标签
plt.title('True labels')
plt.imshow(gt_img / 255.)
plt.subplot(133)  # 显示预测标签
plt.imshow(pred_colored / 255.)
plt.title('predicted labels')