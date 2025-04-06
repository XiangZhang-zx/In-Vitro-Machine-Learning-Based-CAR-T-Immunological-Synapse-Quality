import gdown
import os

# 创建权重存储目录
os.makedirs("pretrained_weights", exist_ok=True)
os.makedirs("postrained_weights", exist_ok=True)

# 下载预训练权重
pretrained_url = "https://drive.google.com/file/d/14S5FM_ToWBW205fZ8gx2NzB5irxilU4r/view?usp=sharing"
gdown.download(pretrained_url, output="pretrained_weights/pretrained.zip", quiet=False)

# 下载后训练权重
postrained_url = "https://drive.google.com/file/d/1ANgtRILhAahkErYWsldRjx1xChyaNZig/view?usp=sharing"
gdown.download(postrained_url, output="postrained_weights/postrained.zip", quiet=False)

# 解压文件
!unzip -q pretrained_weights/pretrained.zip -d pretrained_weights/
!unzip -q postrained_weights/postrained.zip -d postrained_weights/

# 更新P.py文件中的路径
with open("P.py", "r") as f:
    lines = f.readlines()

with open("P.py", "w") as f:
    for line in lines:
        if "pretrained_weights" in line:
            f.write(f'pretrained_weights = "{os.path.abspath("pretrained_weights")}"\n')
        elif "postrained_weights" in line:
            f.write(f'postrained_weights = "{os.path.abspath("postrained_weights")}"\n')
        else:
            f.write(line)

print("权重文件已下载并解压，P.py文件已更新")