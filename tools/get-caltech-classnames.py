import glob
import os
import random

data_root = '/dockerdata/256_ObjectCategories/'
category_folders = sorted(glob.glob(data_root + '*'))

file_write_obj = open("caltech-classnames.txt", 'w')
for categories in category_folders:
    category = categories.split('.')[-1]
    if '-101' in category:
        category = category[:-4]
    if '-' in category:
        category = category.replace('-', ' ')

    file_write_obj.writelines('\'' + category + '\',')
    file_write_obj.write('\n')
file_write_obj.close()



