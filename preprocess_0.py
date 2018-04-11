import hashlib
import os
import shutil

import pandas

input_prefix = '../MURA_trainval/train/'
output_prefix = '../MURA_trainval_keras/'

meta_data = pandas.read_csv('train.csv', header=None).get_values()

os.makedirs(output_prefix, mode=0o700, exist_ok=True)
os.makedirs(output_prefix + '0/', mode=0o700, exist_ok=True)
os.makedirs(output_prefix + '1/', mode=0o700, exist_ok=True)
for x_path, y in meta_data:
    i_path = input_prefix + '/'.join(x_path.split('/')[2:])
    o_path = (output_prefix
              + str(y) + '/'
              + hashlib.sha1(('/'.join(x_path.split('/')[2:])).encode('utf-8')).hexdigest()
              + '.png')
    shutil.copy(i_path, o_path)

input_prefix = '../MURA_trainval/valid/'
output_prefix_1 = '../MURA_valid1_keras/'
output_prefix_2 = '../MURA_valid2_keras/'

meta_data = pandas.read_csv('valid.csv', header=None).get_values()

os.makedirs(output_prefix_1, mode=0o700, exist_ok=True)
os.makedirs(output_prefix_1 + '0/', mode=0o700, exist_ok=True)
os.makedirs(output_prefix_1 + '1/', mode=0o700, exist_ok=True)
os.makedirs(output_prefix_2, mode=0o700, exist_ok=True)
os.makedirs(output_prefix_2 + '0/', mode=0o700, exist_ok=True)
os.makedirs(output_prefix_2 + '1/', mode=0o700, exist_ok=True)
for x_path, y in meta_data:
    # different patient are divided into different sets
    if hashlib.sha1((x_path.split('/')[3]).encode('utf-8')).hexdigest()[0] < '8':
        output_prefix = output_prefix_1
    else:
        output_prefix = output_prefix_2
    i_path = input_prefix + '/'.join(x_path.split('/')[2:])
    o_path = (output_prefix
              + str(y) + '/'
              + hashlib.sha1(('/'.join(x_path.split('/')[2:])).encode('utf-8')).hexdigest()
              + '.png')
    shutil.copy(i_path, o_path)
