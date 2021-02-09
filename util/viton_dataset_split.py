import os
from shutil import copyfile
viton_path = '/home/guopeikun/code/CAGAN/CycleGAN/datasets/viton_dataset'
viton_clothes_path = os.path.join(viton_path, 'train/cloth')
triplet_path = os.path.join(viton_path, 'triplet1')

# mkdir train_x, train_yi, train_yj
train_sets = ['train_x', 'train_yi', 'train_yj']
for dir in train_sets:
    os.makedirs(os.path.join(triplet_path, dir), exist_ok=True)
    print(os.path.join(triplet_path, dir))

cnt = 0
for f in os.listdir(viton_clothes_path):
    cnt += 1
    if cnt%500==0:
        print(str(cnt)+'/7000')
    x_src = os.path.join(viton_path, 'train/image/'+f.replace('_1','_0'))
    y_src = os.path.join(viton_clothes_path, f)
    if cnt<=7000:
        x_dst = os.path.join(triplet_path, train_sets[0], f.replace('_1','_0'))
        yi_dst = os.path.join(triplet_path, train_sets[1], f)
        copyfile(x_src, x_dst)
        copyfile(y_src, yi_dst)
    elif cnt<=14000:
        yj_dst = os.path.join(triplet_path, train_sets[2], f)
        copyfile(y_src, yj_dst)
    else:
        break
