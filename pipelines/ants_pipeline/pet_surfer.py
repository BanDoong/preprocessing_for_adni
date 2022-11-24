import os
import shutil
import numpy as np
import glob
from multiprocessing import Pool
import pandas as pd

DATA_FOLDER = '/media/icml/extremeSSD/free7_RAS'

data_frame = pd.read_csv("/home/icml/Downloads/label_ADNI3_60d.csv")

# SUBJECT_LISTS = data_frame["participant_id"].values
SUBJECT_LISTS = list(data_frame['participant_id'])
ban_list = ["036_S_4715"]
os.environ["SUBJECTS_DIR"] = os.path.abspath(DATA_FOLDER)

def process(sub):
    print(f"Processing: {sub}")
    subject_dir = f'{DATA_FOLDER}/{sub}'
    subsub_file = os.listdir(os.path.join(subject_dir, 'mri'))
    if not os.path.exists(subject_dir):
        return False
    adni3_list = os.listdir('/home/icml/Downloads/adni2testmri/adni3/Nifti')
    adni2_list = os.listdir('/home/icml/Downloads/adni2testmri/nifti_av45')

    FWHM = 8

    ###
    #   Amyloid PET
    ### 
    if 'gtmseg.mgz' not in subsub_file:
        os.system(f'mri_gtmseg --s {sub}')

    # os.system(f'mkdir -p /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/norescale')
    # os.system(
    #     f'mv /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/* /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/norescale')
    # os.system(f'mkdir -p /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/norescale')
    # os.system(
    #     f'mv /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/* /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/norescale')

    pet_folder = f'{subject_dir}/pet_uniform/AV45'
    if not os.path.exists(os.path.join(pet_folder,'pons')):
        if sub in adni3_list and adni2_list:
            pet_file = glob.glob(f'/home/icml/Downloads/adni2testmri/adni3/Nifti/{sub}/*AV45*.nii') + glob.glob(f'/home/icml/Downloads/adni2testmri/adni3/Nifti/{sub}/*FBB*.nii')
        else:
            pet_file = glob.glob(f'/home/icml/Downloads/adni2testmri/nifti_av45/{sub}/*AV45*.nii') + glob.glob(f'/home/icml/Downloads/adni2testmri/nifti_av45/{sub}/*FBB*.nii')

        if len(pet_file) == 0:
            print(f"{sub} is not in here")
            return False

        pet_file = pet_file[0]

        # Create average template.nii.gz
        os.system(f"mri_concat {pet_file} --mean --o {pet_folder}/template.nii.gz > {pet_folder}/log_amyloid.txt")

        # run the rigid (6 DOF) registration
        os.system(f"mri_coreg --s {sub} --mov {pet_folder}/template.nii.gz --reg {pet_folder}/template.reg.lta >> {pet_folder}/log_amyloid.txt")

        # if not os.path.exists(f"/code/ADNI_PVC/{sub}/pet_uniform/AV45"):
        #     os.makedirs(f"/code/ADNI_PVC/{sub}/pet_uniform/AV45")

        # PVC
        os.system(f'mkdir -p /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/pons')
        os.system(f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --psf {FWHM} --seg {subject_dir}/mri/gtmseg.mgz --default-seg-merge --mgx .01 --o /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/pons > /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/pons/log_amyloid_pons.txt")
        os.system(f"mri_convert -at {pet_folder}/template.reg.lta /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/pons/mgx.gm.nii.gz /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/pons/PET_PVC_T1_pons.nii.gz")
        # os.system(f'mv /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/* /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/pons/')

        os.system(f'mkdir -p /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/cerebellum')
        os.system(f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --psf {FWHM} --seg {subject_dir}/mri/gtmseg.mgz --default-seg-merge --mgx .01 --o /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/cerebellum --rescale 7 8 46 47 > /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/cerebellum/log_amyloid_cerebellum.txt")
        os.system(f"mri_convert -at {pet_folder}/template.reg.lta /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/cerebellum/mgx.gm.nii.gz /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/cerebellum/PET_PVC_T1_cerebellum.nii.gz")
        # os.system(f'mv /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/* /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV45/cerebellum/')

        # os.system(f"mri_vol2surf --mov /code/ADNI_PVC/{sub}/pet_uniform/AV45/PET_PVC_T1.nii.gz --regheader {sub} --hemi lh --projfrac 0.5  --o {subject_dir}/surf/lh.amyloid.pvc.fsaverage.nii.gz --cortex --trgsubject fsaverage > {pet_folder}/log_amyloid.txt")
        # os.system(f"mri_vol2surf --mov /code/ADNI_PVC/{sub}/pet_uniform/AV45/PET_PVC_T1.nii.gz --regheader {sub} --hemi rh --projfrac 0.5 --o {subject_dir}/surf/rh.amyloid.pvc.fsaverage.nii.gz --cortex --trgsubject fsaverage >> {pet_folder}/log_amyloid.txt")

        # NOPVC
        # os.system(f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --seg {subject_dir}/mri/gtmseg.mgz --default-seg-merge --no-tfe --auto-mask 1 .01 --o {pet_folder}/nopvc --save-input >> {pet_folder}/log_amyloid.txt")
        # os.system(f"mri_vol2surf --mov {pet_folder}/nopvc/input.rescaled.nii.gz --reg {pet_folder}/nopvc/aux/bbpet2anat.lta --hemi lh --projfrac 0.5  --o {subject_dir}/surf/lh.amyloid.nopvc.fsaverage.0.5.nii.gz --cortex --trgsubject fsaverage >> {pet_folder}/log_amyloid.txt")
        # os.system(f"mri_vol2surf --mov {pet_folder}/nopvc/input.rescaled.nii.gz --reg {pet_folder}/nopvc/aux/bbpet2anat.lta --hemi rh --projfrac 0.5 --o {subject_dir}/surf/rh.amyloid.nopvc.fsaverage.0.5.nii.gz --cortex --trgsubject fsaverage >> {pet_folder}/log_amyloid.txt")

    # ###
    # #   Tau PET
    # ### 
    pet_folder = f'{subject_dir}/pet_uniform/AV1451'
    if not os.path.exists(os.path.join(pet_folder, 'pons')):
        if sub in adni3_list and adni2_list:
            pet_file = glob.glob(f'/home/icml/Downloads/adni2testmri/adni3/Nifti/{sub}/*AV1451*.nii')
        else:
            pet_file = glob.glob(f'/home/icml/Downloads/adni2testmri/nifti_av1451/{sub}/*AV1451*.nii')

        if len(pet_file) == 0:
            print(f"{sub}")
            return False

        pet_file = pet_file[0]

        # # Create average template.nii.gz
        os.system(f"mri_concat {pet_file} --mean --o {pet_folder}/template.nii.gz > {pet_folder}/log_tau.txt")

        # # run the rigid (6 DOF) registration
        os.system(f"mri_coreg --s {sub} --mov {pet_folder}/template.nii.gz --reg {pet_folder}/template.reg.lta >> {pet_folder}/log_tau.txt")

        # if not os.path.exists(f"/code/ADNI_PVC/{sub}/pet_uniform/AV1451"):
        #     os.makedirs(f"/code/ADNI_PVC/{sub}/pet_uniform/AV1451")

        # PVC
        os.system(f'mkdir -p /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/pons')
        os.system(f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --psf {FWHM} --seg {subject_dir}/mri/gtmseg.mgz --default-seg-merge --mgx .01 --o /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/pons  > /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/pons/log_tau_pons.txt")
        os.system(f"mri_convert -at {pet_folder}/template.reg.lta /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/pons/mgx.gm.nii.gz /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/pons/PET_PVC_T1_pons.nii.gz")
        # os.system(
        #     f'mv /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/* /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/pons/')
        os.system(f'mkdir -p /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/cerebellum')
        os.system(
            f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --psf {FWHM} --seg {subject_dir}/mri/gtmseg.mgz --default-seg-merge --mgx .01 --o /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/cerebellum  --rescale 7 8 46 47 > /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/cerebellum/log_tau_cerebellum.txt")
        os.system(
            f"mri_convert -at {pet_folder}/template.reg.lta /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/cerebellum/mgx.gm.nii.gz /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/cerebellum/PET_PVC_T1_cerebellum.nii.gz")
    # os.system(
    #     f'mv /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/* /media/icml/extremeSSD/free7_RAS/{sub}/pet_uniform/AV1451/cerebellum/')


    # os.system(f"mri_vol2surf --mov /code/ADNI_PVC/{sub}/pet_uniform/AV1451/PET_PVC_T1.nii.gz --regheader {sub} --hemi lh --projfrac 0.5  --o {subject_dir}/surf/lh.tau.pvc.fsaverage.nii.gz --cortex --trgsubject fsaverage > {pet_folder}/log_tau.txt")
    # os.system(f"mri_vol2surf --mov /code/ADNI_PVC/{sub}/pet_uniform/AV1451/PET_PVC_T1.nii.gz --regheader {sub} --hemi rh --projfrac 0.5 --o {subject_dir}/surf/rh.tau.pvc.fsaverage.nii.gz --cortex --trgsubject fsaverage >> {pet_folder}/log_tau.txt")

    # NOPVC
    # os.system(f"mri_gtmpvc --i {pet_file} --reg {pet_folder}/template.reg.lta --seg {subject_dir}/mri/gtmseg.mgz --default-seg-merge --no-tfe --auto-mask 1 .01 --o {pet_folder}/nopvc --save-input >> {pet_folder}/log_tau.txt") 
    # os.system(f"mri_vol2surf --mov {pet_folder}/nopvc/input.rescaled.nii.gz --reg {pet_folder}/nopvc/aux/bbpet2anat.lta --hemi lh --projfrac 0.5  --o {subject_dir}/surf/lh.tau.nopvc.fsaverage.0.5.nii.gz --cortex --trgsubject fsaverage >> {pet_folder}/log_tau.txt")
    # os.system(f"mri_vol2surf --mov {pet_folder}/nopvc/input.rescaled.nii.gz --reg {pet_folder}/nopvc/aux/bbpet2anat.lta --hemi rh --projfrac 0.5 --o {subject_dir}/surf/rh.tau.nopvc.fsaverage.0.5.nii.gz --cortex --trgsubject fsaverage >> {pet_folder}/log_tau.txt")

    print(f"Done processing {sub}")

    return True

# process(SUBJECT_LISTS[0])
with Pool(12) as p:
    print(p.map(process, SUBJECT_LISTS))
