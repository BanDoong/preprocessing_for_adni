import os
import ants
from tqdm import tqdm
from ants_coreg import coreg_modality

data_dir = '/media/icml/extremeSSD/free7_RAS'
output_dir = '/home/icml/Desktop/ants_reg_data'
subj_list = os.listdir(data_dir)
subj_list.sort()

subj_list = subj_list[:73]

fix = '/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz'
for subj in tqdm(subj_list):
    mov = os.path.join(data_dir, subj, 'mri/nu.nii.gz')
    out = os.path.join(output_dir, subj, f'{subj}_MRI.nii.gz')
    in_av45 = os.path.join(data_dir, subj, 'pet_uniform/AV45/PET_T1.nii.gz')
    in_av1451 = os.path.join(data_dir, subj, 'pet_uniform/AV1451/PET_T1.nii.gz')
    out_av45 = os.path.join(output_dir, subj, f'{subj}_Amyloid.nii.gz')
    out_av1451 = os.path.join(output_dir, subj, f'{subj}_Tau.nii.gz')
    coreg_modality(maksOrNot='img', fix=fix, mov=mov, out=out, in_av45=in_av45, out_av45=out_av45, in_av1451=in_av1451,
                   out_av1451=out_av1451)
