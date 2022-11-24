import ants
import os
import argparse

parser = argparse.ArgumentParser("REGISTRATION FOR ROI")
parser.add_argument('--path', default='/home/id202188508/all/freesurfer_RAS/adni3/free7_RAS/')
parser.add_argument('--fix', default='/home/id202188508/tpl-MNI152NLin2009cSym/MNI152_T1_1mm_brain.nii.gz')

args = parser.parse_args()

subj_list = os.listdir(args.path)
subj_list.sort()


def coreg_modality(fix=None, mov=None, out=None):
    fix = ants.image_read(fix)  # reference path
    mov = ants.image_read(mov)  # T1 path
    reg_output = ants.registration(fix, mov, type_of_transforme='SyN')  # matrix
    ants.image_write(reg_output['warpedmovout'], out)  # out : t1 output

if 'adni1' in args.path:
    ban_list = ['100_S_0747']
else:
    ban_list = []
for subj in subj_list:
    if subj in ban_list:
        print(f"{subj} is in ban_list")
    else:
        print(f'{subj} is registering')
        total_path = os.path.join(args.path, subj, 'mri/aparc+aseg.nii.gz')
        output_path = os.path.join(args.path, subj, f'reg_label')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        coreg_modality(args.fix, total_path, os.path.join(output_path, f'{subj}_reg.nii.gz'))
