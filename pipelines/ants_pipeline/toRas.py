import os

"""
## change name
path = '/home/icml/Desktop/RawData'
subj_list = os.listdir(path)
subj_list.sort()
i = 0
j = 0
for subj in subj_list:
    sub_dir_path = os.path.join(path, subj)
    sub_data = os.listdir(sub_dir_path)
    for sub in sub_data:
        if ('MPRAGE' in sub or 'ADNI' in sub or 'Acc' in sub) and 'nii' in sub:
            # os.system(f'cp {sub_dir_path}/{sub_data} {subj}_mri.nii')
            os.system(f'cp {sub_dir_path}/{sub} {sub_dir_path}/{subj}_mri.nii')
        else:
            i += 1

    # if i == 6:
    #     i = 0
print(i)
"""

# path = '/home/icml/Desktop/'
#
# for i in range(29):
#     # os.system(f'mkdir /home/icml/Desktop/split_data/data_{i + 1} ')
#     os.system(
#         f'ls -1 /home/icml/Desktop/moving | head -25 | xargs -i mv /home/icml/Desktop/moving/"{}" /home/icml/Desktop/split_data/data_{i + 1}')
# #
# subj_list = os.listdir(path)

# for subj in subj_list:
#     os.system('sh /home/icml/Downloads/Brain_surface-master nii2ras.sh')

import argparse
parser = argparse.ArgumentParser(description='TO RAS Coordinate')
parser.add_argument('--path', type=str, default='/home/icml/Downloads/adni2testmri/nifti')
args = parser.parse_args()

path = args.path
subj_list = os.listdir(path)
subj_list.sort()

for subj in subj_list:
    sub_dir_path = os.path.join(path, subj)
    sub_data = os.listdir(sub_dir_path)
    for sub in sub_data:
        if 'nii' in sub and not 'mri' in sub:
            os.system(f'cp {sub_dir_path}/{sub} {sub_dir_path}/{subj}_mri.nii')

os.system('sh /home/icml/Downloads/Brain_surface-master/nii2ras.sh')


# for subj in subj_list:
#     os.system('sh /home/icml/Downloads/Brain_surface-master nii2ras.sh')
