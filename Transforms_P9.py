from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python的用法-》 tensor数据类型
# 通过 transforms.Totensor去看两个问题
# 1、 transforms 该如何使用（python）
# 2、 为什么我们需要Tensor数据类型

img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
writer.add_image("image", tensor_img)
writer.close()