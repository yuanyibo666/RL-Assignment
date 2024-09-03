from PIL import Image
import os

# 设置源文件夹和目标文件夹
src_folder = r'C:\Users\yuanyibo\Desktop\Reinforce\assignment\code\assignment1\result'
dst_folder = r'C:\Users\yuanyibo\Desktop\Reinforce\assignment\code\assignment1\result\jpg'

# 确保目标文件夹存在
os.makedirs(dst_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(src_folder):
    if filename.endswith(".png"):
        # 构造完整的文件路径
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename[:-4] + ".jpg")
        
        # 打开并转换图片
        with Image.open(src_path) as img:
            img = img.convert('RGB')  # 转换为RGB模式，以确保没有透明度
            img.save(dst_path, 'JPEG')

print("转换完成。")