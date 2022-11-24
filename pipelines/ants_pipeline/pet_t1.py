import os
import argparse
import multiprocessing

# path = 'caps/subjects'
# subj_list = os.listdir(path)
# raw_path = 'raw_data'
#
#
# def get_subdir(path, subj):
#     subj_dir = os.path.join(path, subj)
#     ses = os.listdir(subj_dir)
#     for s in ses:
#         if 'ses' in s:
#             ses_dir = os.path.join(subj_dir, s)
#             return ses_dir, s
parser = argparse.ArgumentParser('PET to MRI space')
parser.add_argument('--input', default='/home/icml/Downloads/ADNI2/av45_nifti')
parser.add_argument('--output', default='/home/icml/Desktop/free7_RAS')
args = parser.parse_args()
path = args.input
mri_ = args.output

os.system('export FREESURFER_HOME="/usr/local/freesurfer/7.2.0"')
os.system(f'export SUBJECTS_DIR={mri_}')
os.system(f'source $FREESURFER_HOME/SetUpFreeSurfer.sh')
os.system('echo $FREESURFER_HOME')
os.system('echo $SUBJECTS_DIR')
if 'av45' in path:
    pet_dir = 'AV45'
elif 'av1451' in path:
    pet_dir = 'AV1451'
else:
    pet_dir = 'FDG'

subj_list = os.listdir(path)
subj_list.sort()


def run_(subj):
    # for subj in subj_list:
    # ses_dir, s = get_subdir(path, subj)
    # transform_subj_name = subj[8:11] + '_S_' + subj[12:16]
    mri_path = os.path.join(mri_, subj)
    pet_path = os.path.join(path, subj)
    pet_sub_list = os.listdir(pet_path)
    # change
    # pet_dir = 'AV45'
    os.system(f'mkdir -p {mri_}/{subj}/pet_uniform/{pet_dir}')
    # mri_path = os.path.join(ses_dir, 't1_linear', f'{subj}_{s}__T1w_space-MNI152NLin2009cSym_res-1x1x1_T1w.nii.gz')
    # pet_path = os.path.join(ses_dir, 'pet_lienar')
    # pet_sub_list = os.listdir(os.path.join(raw_path, transform_subj_name))
    for sub_list in pet_sub_list:
        if pet_dir == 'AV45':
            if ('AV45' or 'FBB' in sub_list) and 'nii' in sub_list:
                av45 = os.path.join(pet_path, sub_list)
                os.system(
                    f'mri_coreg --s {subj} --mov {av45} --reg {mri_}/{subj}/pet_uniform/{pet_dir}/template.reg.lta')
                os.system(
                    f'mri_convert -at {mri_}/{subj}/pet_uniform/{pet_dir}/template.reg.lta {av45} {mri_}/{subj}/pet_uniform/{pet_dir}/PET_T1.nii.gz')
                os.system(f'mri_convert {mri_}/{subj}/mri/nu.mgz {mri_}/{subj}/pet_uniform/{pet_dir}/T1.nii.gz')
        elif pet_dir == 'AV1451':
            if 'AV1451' in sub_list and 'nii' in sub_list:
                av1451 = os.path.join(pet_path, sub_list)
                os.system(
                    f'mri_coreg --s {subj} --mov {av1451} --reg {mri_}/{subj}/pet_uniform/{pet_dir}/template.reg.lta')
                os.system(
                    f'mri_convert -at {mri_}/{subj}/pet_uniform/{pet_dir}/template.reg.lta {av1451} {mri_}/{subj}/pet_uniform/{pet_dir}/PET_T1.nii.gz')
                os.system(f'mri_convert {mri_}/{subj}/mri/nu.mgz {mri_}/{subj}/pet_uniform/{pet_dir}/T1.nii.gz')
        else:
            if 'fdg' in sub_list and 'nii' in sub_list:
                av1451 = os.path.join(pet_path, sub_list)
                os.system(
                    f'mri_coreg --s {subj} --mov {av1451} --reg {mri_}/{subj}/pet_uniform/{pet_dir}/template.reg.lta')
                os.system(
                    f'mri_convert -at {mri_}/{subj}/pet_uniform/{pet_dir}/template.reg.lta {av1451} {mri_}/{subj}/pet_uniform/{pet_dir}/PET_T1.nii.gz')
                os.system(f'mri_convert {mri_}/{subj}/mri/nu.mgz {mri_}/{subj}/pet_uniform/{pet_dir}/T1.nii.gz')
    # os.system(f'mri_coreg --mov {av1451} --ref {mri_path} --reg {pet_path}_tau.lta')
    # os.system(f'mri_convert -at {pet_path}_amyloid.lta {av45} {pet_path}/{subj}_{s}_av45.nii.gz')
    # os.system(f'mri_convert -at {pet_path}_tau.lta {av1451} {pet_path}/{subj}_{s}_av1451.nii.gz')


with multiprocessing.Pool(10) as p:
    p.map(run_, subj_list)
