import os
import ants
import pandas as pd
from tqdm import tqdm
from ants_coreg import coreg_modality
import multiprocessing
import argparse

parser = argparse.ArgumentParser('ANTS COREGISTRATION')
parser.add_argument('--data_dir', default='/home/icml/Downloads/ADNI1/free7_RAS')
parser.add_argument('--fix', default='/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz')
parser.add_argument('--output_dir', default='/media/icml/extremeSSD/Data_ADNI1_ANTS')
# data_dir = '/media/icml/extremeSSD/free7_RAS'
# output_dir = '/home/icml/Desktop/ants_reg_data'
# output_dir = '/home/icml/Downloads/ADNI1/Data_ADNI1_ANTS'
args = parser.parse_args()
data_dir = args.data_dir
output_dir = args.output_dir
fix = args.fix
# subj_list = os.listdir(data_dir)
subj_list = list(pd.read_csv("/home/icml/Downloads/label_ADNI3_60d.csv")['participant_id'])
subj_list.sort()

# adni2 adding
# ban_list = ['019_S_5242', '037_S_0303', '037_S_0467']
ban_list = ['036_S_4715']


def run(subj):
    print(f"{subj} is coreg by ANTS")
    if not os.path.exists(os.path.join(output_dir, subj)):
        os.makedirs(os.path.join(output_dir, subj))
    if subj not in ban_list:
        # os.system(
        #     f'mri_convert {os.path.join(data_dir, subj, "mri/nu.mgz")} {os.path.join(data_dir, subj, "mri/nu.nii.gz")} > /dev/null')

        mov = os.path.join(data_dir, subj, 'mri/nu.nii.gz')
        out = os.path.join(output_dir, subj, f'{subj}_MRI.nii.gz')

        in_av45 = os.path.join(data_dir, subj, 'pet_uniform/AV45/PET_PVC_T1.nii.gz')
        out_av45 = os.path.join(output_dir, subj, f'{subj}_Amyloid.nii.gz')
        # in_av1451 = os.path.join(data_dir, subj, 'pet_uniform/AV1451/PET_T1.nii.gz')
        # out_av1451 = os.path.join(output_dir, subj, f'{subj}_Tau.nii.gz')
        coreg_modality(maksOrNot='img', fix=fix, mov=mov, out=out, in_av45=in_av45, out_av45=out_av45)
        in_av1451 = os.path.join(data_dir, subj, 'pet_uniform/AV1451/PET_PVC_T1.nii.gz')
        out_av1451 = os.path.join(output_dir, subj, f'{subj}_Tau.nii.gz')
        coreg_modality(maksOrNot='img', fix=fix, mov=mov, out=out, in_av45=in_av1451, out_av45=out_av1451)


# with multiprocessing.Pool(5) as p:
#     p.map(run, subj_list)
# print(subj_list[324])
# subj_list = subj_list[325:]
for subj in subj_list:
    if subj not in ban_list:
        run(subj)
        os.system('rm -f /tmp/* > /dev/null')

print("FINISH")
