# This file should be located in the 256_ObjectCategories/ folder in Caltech-256 dataset which has subfolders like "001.ak47"

import glob
import os
import random

category_folders = sorted(glob.glob('*'))
print(category_folders)

m = 60  # shot number
cls = 256  # class number
total = m*cls/1000
m_shot_train_path_name = '_train_'+str(cls)+'cls_' +str(m)+'shot_'+str(int(total))+'k'
os.system('mkdir ../%s' % m_shot_train_path_name)
m_shot_val_path_name = '_val_'+str(cls)+'cls_' +str(m)+'shot_'+str(int(total))+'k'
os.system('mkdir ../%s' % m_shot_val_path_name)

# random.shuffle(category_folders)
category_folders = category_folders[:cls]

for category in category_folders:

    category_files = glob.glob('%s/*' % category)
    random.shuffle(category_files)
    train_files = category_files[:m]
    val_files = category_files[m:]
    os.system('mkdir ../{}/{}'.format(m_shot_train_path_name, category))
    os.system('mkdir ../{}/{}'.format(m_shot_val_path_name, category))
    # import ipdb
    # ipdb.set_trace(context=20)
    for train_file in train_files:
        os.system('cp {} ../{}/{}/'.format(train_file, m_shot_train_path_name, category))
        # print('cp {} ../{}/{}/'.format(copy_files[i], m_shot_path_name, category))
    for val_file in val_files:
        os.system('cp {} ../{}/{}/'.format(val_file, m_shot_val_path_name, category))

os.system('mkdir ../../{}'.format("caltech256"+m_shot_train_path_name))
os.system('mkdir ../../{}/train'.format("caltech256"+m_shot_train_path_name))
os.system('mv ../{}/* ../../{}/train/'.format(m_shot_train_path_name, "caltech256"+m_shot_train_path_name))

# os.system('mkdir ../../{}'.format("caltech256"+m_shot_val_path_name))
os.system('mkdir ../../{}/val'.format("caltech256"+m_shot_train_path_name))
os.system('mv ../{}/* ../../{}/val/'.format(m_shot_val_path_name, "caltech256"+m_shot_train_path_name))
# os.system('cp -r ../val ../../{}/'.format("caltech256_img_train"+m_shot_path_name))