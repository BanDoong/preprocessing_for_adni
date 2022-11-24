import os
from tqdm import tqdm
import nibabel as nib
import numpy as np
import skimage.transform as skTrans


# pet to t1(freesurfer) => fsl coreg

def Norm(img_mri, img_tau, img_amyloid):
    img_mri = (img_mri - np.min(img_mri)) / (np.max(img_mri) - np.min(img_mri))
    img_tau = (img_tau - np.min(img_tau)) / (np.max(img_tau) - np.min(img_tau))
    img_amyloid = (img_amyloid - np.min(img_amyloid)) / (np.max(img_amyloid) - np.min(img_amyloid))

    return img_mri, img_tau, img_amyloid


def resizing_norm(path, subj_list, image_size):
    ban_list = ['009_S_6212', '016_S_6809', '027_S_5079', '027_S_5109', '027_S_6463', '057_S_6869', '070_S_6911',
                '099_S_6038', '126_S_0680', '129_S_6784', '023_S_6334']
    for subj in tqdm(subj_list):
        if subj in ban_list:
            pass
        else:
            print(subj)
            mri = os.path.join(path, subj, subj + "_MRI_mask_norm_cropped.nii.gz")
            tau = os.path.join(path, subj, subj + "_Tau_mask_norm_cropped.nii.gz")
            amyloid = os.path.join(path, subj, subj + "_Amyloid_mask_norm_cropped.nii.gz")
            img_mri = nib.load(mri)
            mri_header = img_mri.header
            mri_data = np.array(img_mri.dataobj)
            img_tau = nib.load(tau)
            tau_header = img_tau.header
            tau_data = np.array(img_tau.dataobj)
            img_amyloid = nib.load(amyloid)
            amyloid_header = img_amyloid.header
            amyloid_data = np.array(img_amyloid.dataobj)
            mri_data = skTrans.resize(mri_data, image_size, order=1, preserve_range=True)
            tau_data = skTrans.resize(tau_data, image_size, order=1, preserve_range=True)
            amyloid_data = skTrans.resize(amyloid_data, image_size, order=1, preserve_range=True)
            img_mri, img_tau, img_amyloid = Norm(mri_data, tau_data, amyloid_data)
            img_mri = nib.Nifti1Image(img_mri, affine=mri_header.get_best_affine(), header=mri_header)
            img_tau = nib.Nifti1Image(img_tau, affine=tau_header.get_best_affine(), header=tau_header)
            img_amyloid = nib.Nifti1Image(img_amyloid, affine=amyloid_header.get_best_affine(), header=amyloid_header)
            nib.save(img_mri, mri[:-7] + '_resized.nii.gz')
            nib.save(img_tau, tau[:-7] + '_resized.nii.gz')
            nib.save(img_amyloid, amyloid[:-7] + '_resized.nii.gz')


def make_mask(path, subj_list):
    """
    making mask
    """


def get_subdir(path, subj):
    subj_dir = os.path.join(path, subj)
    ses = os.listdir(subj_dir)
    for s in ses:
        if 'ses' in s:
            ses_dir = os.path.join(subj_dir, s)
            return ses_dir, s


path = 'caps/subjects'
subj_list = os.listdir(path)
raw_path = 'raw_data'


def pet2t1(path, subj_list):
    for subj in subj_list:
        ses_dir, s = get_subdir(path, subj)
        transform_subj_name = subj[8:11] + '_S_' + subj[12:16]
        # change
        mri_path = os.path.join(path, subj, f'{subj}_MRI.nii.gz')
        pet_path = os.path.join(path, subj)
        pet_sub_list = os.listdir(os.path.join(raw_path, transform_subj_name))
        for sub_list in pet_sub_list:
            if 'AV45' or 'FBB' in sub_list and 'nii' in sub_list:
                av45 = os.path.join(raw_path, transform_subj_name, sub_list)
            if 'AV1451' in sub_list and 'nii' in sub_list:
                av1451 = os.path.join(raw_path, transform_subj_name, sub_list)

        os.system(f'mri_coreg --mov {av45} --ref {mri_path} --reg {pet_path}_amyloid.lta')
        os.system(f'mri_coreg --mov {av1451} --ref {mri_path} --reg {pet_path}_tau.lta')
        os.system(f'mri_convert -at {pet_path}_amyloid.lta {av45} {pet_path}/{subj}_{s}_av45.nii.gz')
        os.system(f'mri_convert -at {pet_path}_tau.lta {av1451} {pet_path}/{subj}_{s}_av1451.nii.gz')


def coreg_fsl(list_subj):
    print(f'subjects Number == {len(list_subj)}')
    data_dir = '/media/icml/extremeSSD/free7_RAS'
    output_dir = '/media/icml/extremeSSD/Data_FSL'
    for subj in list_subj:
        print(subj)
        mri_path = os.path.join(data_dir, subj, 'mri/nu.nii.gz')
        av1451_path = os.path.join(data_dir, subj, 'pet_uniform/AV1451/PET_T1.nii.gz')
        av45_path = os.path.join(data_dir, subj, 'pet_uniform/AV45/PET_T1.nii.gz')
        print(
            f'/usr/local/fsl/bin/flirt -in {mri_path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz -out {os.path.join(output_dir, subj)}/{subj}_MRI.nii.gz -omat {os.path.join(output_dir, subj)}/{subj}_MRI.mat -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear')
        print(
            f'/usr/local/fsl/bin/flirt -in {av45_path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz -applyxfm -init {os.path.join(output_dir, subj)}/{subj}_MRI.mat -omat {os.path.join(output_dir, subj)}/{subj}_Amyloid.mat -out {os.path.join(output_dir, subj)}/{subj}_Amyloid.nii.gz -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear')
        print(
            f'/usr/local/fsl/bin/flirt -in {av1451_path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz -applyxfm -init {os.path.join(output_dir, subj)}/{subj}_MRI.mat -omat {os.path.join(output_dir, subj)}/{subj}_Tau.mat -out {os.path.join(output_dir, subj)}/{subj}_Tau.nii.gz -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear')

        # os.system(
        #     f'/usr/local/fsl/bin/flirt -in {mri_path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz -out {os.path.join(output_dir, subj)}/{subj}_MRI.nii.gz -omat {os.path.join(output_dir, subj)}/{subj}_MRI.mat -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear')
        # os.system(
        #     f'/usr/local/fsl/bin/flirt -in {av45_path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz -applyxfm -init {os.path.join(output_dir, subj)}/{subj}_MRI.mat -omat {os.path.join(output_dir, subj)}/{subj}_Amyloid.mat -out {os.path.join(output_dir, subj)}/{subj}_Amyloid.nii.gz -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear')
        # os.system(
        #     f'/usr/local/fsl/bin/flirt -in {av1451_path} -ref /usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz -applyxfm -init {os.path.join(output_dir, subj)}/{subj}_MRI.mat -omat {os.path.join(output_dir, subj)}/{subj}_Tau.mat -out {os.path.join(output_dir, subj)}/{subj}_Tau.nii.gz -bins 256 -cost corratio -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12  -interp trilinear')
