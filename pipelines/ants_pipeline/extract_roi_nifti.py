import os
import argparse

parser = argparse.ArgumentParser("NIFTI FOR ROI")
parser.add_argument('--path', default='/home/id202188508/all/freesurfer_RAS/adni3/free7_RAS')
parser.add_argument('--fix', default='/home/id202188508/tpl-MNI152NLin2009cSym/MNI152_T1_1mm_brain.nii.gz')

args = parser.parse_args()

subj_list = os.listdir(args.path)
subj_list.sort()
for subj in subj_list:
    print(f'{subj} mgz --> nifti')
    total_path = os.path.join(args.path, subj, 'mri/aparc+aseg.mgz')
    os.system(f'mri_convert {total_path} {os.path.join(args.path, subj, "mri/aparc+aseg.nii.gz")}')
