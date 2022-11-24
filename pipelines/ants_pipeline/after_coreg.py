import scipy.ndimage
import numpy as np
import nibabel as nib
import os
import skimage.transform as skTrans
import argparse
import scipy.ndimage as ndi
from torchio import RandomNoise
import random
import multiprocessing
import pandas as pd
np.random.seed(0)
random.seed(0)


def augment_image(img, rotate, shift, flip):
    # pdb.set_trace()
    img = scipy.ndimage.interpolation.rotate(img, rotate[0], axes=(1, 0), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[1], axes=(0, 2), reshape=False)
    img = scipy.ndimage.interpolation.rotate(img, rotate[2], axes=(1, 2), reshape=False)
    img = scipy.ndimage.shift(img, shift[0])
    # img = random_noise(img)
    if flip[0] == 1:
        img = np.flip(img, 0) - np.zeros_like(img)
    return img


def random_noise(img):
    mean = np.mean(img)
    std = np.std(img)
    noise = np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * std / 5 + mean
    img = img + noise
    return img


# from torchio import RandomBlur
# def gaussian_blur(img, kernel_size=int(32 * 0.1)):
#     stds_channels = np.tile(stds, (image.num_channels, 1))
#     std_physical = np.array(std_voxel) / np.array(spacing)
#     blurred = ndi.gaussian_filter(data, std_physical)


def crop_img(path, subj, modality):
    nifti = nib.load(path)
    data = np.array(nifti.dataobj)
    # aff = nifti.header
    non_zero = np.nonzero(data)
    x_min = non_zero[0].min()
    x_max = non_zero[0].max()
    y_min = non_zero[1].min()
    y_max = non_zero[1].max()
    z_min = non_zero[2].min()
    z_max = non_zero[2].max()

    x_h = 160 // 2
    y_h = 192 // 2
    z_h = 160 // 2

    x_mid = int(np.round(np.median([x_min, x_max])))
    y_mid = int(np.round(np.median([y_min, y_max])))
    z_mid = int(np.round(np.median([z_min, z_max])))

    crop = data[x_mid - x_h:x_mid + x_h, y_mid - y_h:y_mid + y_h, z_mid - z_h + 2:z_mid + z_h + 2]

    nifti = nib.Nifti1Image(crop, np.eye(4))
    nib.save(nifti, f'/media/icml/extremeSSD/Data_FSL/{subj}/{subj}_{modality}_mask_norm_cropped.nii.gz')


def crop_in_numpy(img):
    non_zero = np.nonzero(img)
    x_min = non_zero[0].min()
    x_max = non_zero[0].max()
    y_min = non_zero[1].min()
    y_max = non_zero[1].max()
    z_min = non_zero[2].min()
    z_max = non_zero[2].max()
    x_h = 160 // 2
    y_h = 192 // 2
    z_h = 160 // 2
    x_mid = int(np.round(np.median([x_min, x_max])))
    y_mid = int(np.round(np.median([y_min, y_max])))
    z_mid = int(np.round(np.median([z_min, z_max])))

    crop_img = img[x_mid - x_h:x_mid + x_h, y_mid - y_h:y_mid + y_h, z_mid - z_h + 2:z_mid + z_h + 2]
    return crop_img


def norm(img_1):
    img_1 = (img_1 - np.mean(img_1)) / np.std(img_1)
    return img_1


def crop_in_fsl(img_path):
    center_x, center_y, center_z = 182 / 2, 218 / 2, 182 / 2
    crop_x, crop_y, crop_z = 160, 192, 160
    img = nib.load(img_path).get_fdata()
    non_zero = np.nonzero(img)
    x_min = non_zero[0].min()
    y_min = non_zero[1].min()
    z_min = non_zero[2].min()
    # os.system(f'mri_convert --crop {center_x} {center_y} {center_z} '
    #           f'--cropsize {crop_x} {crop_y} {crop_z} {img_path} {img_path[:-7] + f"_crop.nii.gz"}')
    os.system(
        f'fslroi {img_path} {img_path[:-7] + f"_crop_fsl.nii.gz"} {x_min} {crop_x} {y_min} {crop_y} {z_min} {crop_z} > /dev/null')


def crop_in_mri_convert(img_path):
    center_x, center_y, center_z = int(182 / 2), int(218 / 2), int(182 / 2)
    crop_x, crop_y, crop_z = 160, 192, 160
    os.system(
        f'mri_convert --crop {center_x} {center_y} {center_z} --cropsize {crop_x} {crop_y} {crop_z} {img_path} {img_path[:-7] + f"_crop.nii.gz"} > /dev/null')


def min_max_norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


##
# parser = argparse.ArgumentParser(description="After Coregistration")
# parser.add_argument('--path', default='/home/icml/Desktop/Data_adni_1', type=str)
# args = parser.parse_args()
# path = args.path
# subj_id = os.listdir(path)
# subj_id.sort()
##
############### ADNI3 ###############
# modality = ['_Amyloid_mask.nii.gz', '_Tau_mask.nii.gz', '_MRI_mask.nii.gz']
# ban_list = ['009_S_6212', '012_S_6073', '016_S_6809', '027_S_5079', '027_S_6001', '027_S_6842', '027_S_5109',
#             '027_S_6463', '057_S_6869', '070_S_6911', '029_S_6798', '035_S_4464', '041_S_4510',
#             '099_S_6038', '126_S_0680', '129_S_6784', '023_S_6334', '094_S_6278', '114_S_6524']
# subj_id = ['009_S_6212', '012_S_6073', '016_S_6809', '027_S_5079', '027_S_6001', '027_S_6842', '027_S_5109',
#             '027_S_6463', '057_S_6869', '070_S_6911', '029_S_6798', '035_S_4464', '041_S_4510',
#             '099_S_6038', '126_S_0680', '129_S_6784', '023_S_6334', '094_S_6278', '114_S_6524']
# ban_list = []
#####################################

############### ADNI1 ###############
# modality = ['_fdg_mask.nii.gz', '', '_MRI_mask.nii.gz']
# ban_list = ['100_S_0747', '031_S_0830']
#####################################

# aug 1,2 : rotation shift flip
# aug 3,4 : rotation shift flip Random noise
aug_size = 10

""" resize,crop, augmentation image"""

# def run_resize(subj):
#     if subj in ban_list:
#         print(f'{subj} is passed')
#         pass
#     else:
#         print(f'{subj} is preprocessing')
#         # os.system(
#         #     f'mri_mask {os.path.join(path, subj, f"{subj}_MRI.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(path, subj)}/{subj}_MRI_mask.nii.gz > /dev/null')
#         # os.system(
#         #     f'mri_mask {os.path.join(path, subj, f"{subj}_Amyloid.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(path, subj)}/{subj}_Amyloid_mask.nii.gz > /dev/null')
#         # os.system(
#         #     f'mri_mask {os.path.join(path, subj, f"{subj}_Tau.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(path, subj)}/{subj}_Tau_mask.nii.gz > /dev/null')
#         # amyloid = os.path.join(path, subj, f'{subj}{modality[0]}')
#         # tau = os.path.join(path, subj, f'{subj}{modality[1]}')
#         # mri = os.path.join(path, subj, f'{subj}{modality[2]}')
#         #
#         # crop_in_mri_convert(mri)
#         # crop_in_mri_convert(amyloid)
#         # crop_in_mri_convert(tau)
#         # crop_in_fsl(mri)
#         # crop_in_fsl(amyloid)
#         # crop_in_fsl(tau)
#
#         img_amyloid = os.path.join(path, subj, f'{subj}{modality[0]}')[:-7] + '_crop.nii.gz'
#         # img_tau = os.path.join(path, subj, f'{subj}{modality[1]}')[:-7] + '_crop.nii.gz'
#         img_mri = os.path.join(path, subj, f'{subj}{modality[2]}')[:-7] + '_crop.nii.gz'
#
#         img_mri = nib.load(img_mri).get_fdata()
#         img_amyloid = nib.load(img_amyloid).get_fdata()
#         # img_tau = nib.load(img_tau).get_fdata()
#
#         img_mri = skTrans.resize(img_mri, (80, 96, 80), order=1, preserve_range=True)
#         # img_tau = skTrans.resize(img_tau, (80, 96, 80), order=1, preserve_range=True)
#         img_amyloid = skTrans.resize(img_amyloid, (80, 96, 80), order=1, preserve_range=True)
#
#         rotate_list = np.random.uniform(-2, 2, (aug_size - 1, 3))
#         shift_list = np.random.uniform(-2, 2, (aug_size - 1, 1))
#         flip_list = np.random.randint(0, 2, (aug_size - -1, 1))
#
#         # img_tau = norm(img_tau)
#         img_amyloid = norm(img_amyloid)
#         img_mri = norm(img_mri)
#
#         img_mri_list = [img_mri]
#         img_amyloid_list = [img_amyloid]
#         # img_tau_list = [img_tau]
#
#         # img_mri = min_max_norm(img_mri)
#         # img_amyloid = min_max_norm(img_amyloid)
#         for j in range(aug_size - 1):
#             img_amyloid_list.append(augment_image(img_amyloid, rotate_list[j], shift_list[j], flip_list[j]))
#             # img_tau_list.append(augment_image(img_tau, rotate_list[j], shift_list[j], flip_list[j]))
#             img_mri_list.append(augment_image(img_mri, rotate_list[j], shift_list[j], flip_list[j]))
#         img_amyloid_list = np.stack(img_amyloid_list, 0)
#         # img_tau_list = np.stack(img_tau_list, 0)
#         img_mri_list = np.stack(img_mri_list, 0)
#
#         for k in range(img_mri_list.shape[0]):
#             img_mri = nib.Nifti1Image(img_mri_list[k], np.eye(4))
#             nib.save(img_mri, os.path.join(path, subj, f'{subj}_MRI_mask_norm_crop_resize_aug_{k}.nii.gz'))
#             img_amyloid = nib.Nifti1Image(img_amyloid_list[k], np.eye(4))
#             nib.save(img_amyloid, os.path.join(path, subj, f'{subj}_Amyloid_mask_norm_crop_resize_aug_{k}.nii.gz'))
#             # img_tau = nib.Nifti1Image(img_tau_list[k], np.eye(4))
#             # nib.save(img_tau, os.path.join(path, subj, f'{subj}_Tau_mask_norm_crop_resize_aug_{k}.nii.gz'))


""" Crop image"""


def run_crop_adni3(subj):
    adni3_path = '/home/icml/Desktop/Data_ANTS_PVC'
    modality = ['_Amyloid_mask.nii.gz', '_Tau_mask.nii.gz', '_MRI_mask.nii.gz']
    ban_list = ['036_S_4715']
    if subj in ban_list:
        print(f'{subj} is passed')
        pass
    else:
        print(f'{subj} is preprocessing')
        os.system(
            f'mri_mask {os.path.join(adni3_path, subj, f"{subj}_MRI.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(adni3_path, subj)}/{subj}_MRI_mask.nii.gz > /dev/null')
        os.system(
            f'mri_mask {os.path.join(adni3_path, subj, f"{subj}_Amyloid.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(adni3_path, subj)}/{subj}_Amyloid_mask.nii.gz > /dev/null')
        os.system(
            f'mri_mask {os.path.join(adni3_path, subj, f"{subj}_Tau.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(adni3_path, subj)}/{subj}_Tau_mask.nii.gz > /dev/null')

        amyloid = os.path.join(adni3_path, subj, f'{subj}{modality[0]}')
        tau = os.path.join(adni3_path, subj, f'{subj}{modality[1]}')
        mri = os.path.join(adni3_path, subj, f'{subj}{modality[2]}')

        crop_in_mri_convert(mri)
        crop_in_mri_convert(amyloid)
        crop_in_mri_convert(tau)
        # crop_in_fsl(mri)
        # crop_in_fsl(amyloid)
        # crop_in_fsl(tau)

        img_amyloid = os.path.join(adni3_path, subj, f'{subj}{modality[0]}')[:-7] + '_crop.nii.gz'
        img_tau = os.path.join(adni3_path, subj, f'{subj}{modality[1]}')[:-7] + '_crop.nii.gz'
        img_mri = os.path.join(adni3_path, subj, f'{subj}{modality[2]}')[:-7] + '_crop.nii.gz'

        img_mri = nib.load(img_mri).get_fdata()
        img_tau = nib.load(img_tau).get_fdata()
        img_amyloid = nib.load(img_amyloid).get_fdata()

        img_mri = norm(img_mri)
        img_tau = norm(img_tau)
        img_amyloid = norm(img_amyloid)

        img_mri = nib.Nifti1Image(img_mri, np.eye(4))
        nib.save(img_mri, os.path.join(adni3_path, subj, f'{subj}_MRI_mask_norm_crop.nii.gz'))
        img_amyloid = nib.Nifti1Image(img_amyloid, np.eye(4))
        nib.save(img_amyloid, os.path.join(adni3_path, subj, f'{subj}_Amyloid_mask_norm_crop.nii.gz'))
        img_tau = nib.Nifti1Image(img_tau, np.eye(4))
        nib.save(img_tau, os.path.join(adni3_path, subj, f'{subj}_Tau_mask_norm_crop.nii.gz'))


def run_crop_adni1(subj):
    adni1_path = '/home/id202188508/all/Data_ADNI1_ANTS'
    modality = ['_fdg_mask.nii.gz', '', '_MRI_mask.nii.gz']
    ban_list = ['100_S_0747', '031_S_0830']

    if subj in ban_list:
        print(f'{subj} is passed')
        pass
    else:
        print(f'{subj} is preprocessing')
        # os.system(
        #     f'mri_mask {os.path.join(path, subj, f"{subj}_MRI.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(path, subj)}/{subj}_MRI_mask.nii.gz > /dev/null')
        # os.system(
        #     f'mri_mask {os.path.join(path, subj, f"{subj}_fdg.nii.gz")} /usr/local/fsl/data/standard/MNI152_T1_1mm_brain_mask_dil.nii.gz {os.path.join(path, subj)}/{subj}_fdg_mask.nii.gz > /dev/null')

        # amyloid = os.path.join(path, subj, f'{subj}{modality[0]}')
        # tau = os.path.join(path, subj, f'{subj}{modality[1]}')
        # mri = os.path.join(path, subj, f'{subj}{modality[2]}')
        #
        # crop_in_mri_convert(mri)
        # crop_in_mri_convert(amyloid)
        # crop_in_mri_convert(tau)
        # crop_in_fsl(mri)
        # crop_in_fsl(amyloid)
        # crop_in_fsl(tau)

        img_fdg = os.path.join(adni1_path, subj, f'{subj}{modality[0]}')[:-7] + '_crop_fsl.nii.gz'
        img_mri = os.path.join(adni1_path, subj, f'{subj}{modality[2]}')[:-7] + '_crop_fsl.nii.gz'

        img_mri = nib.load(img_mri).get_fdata()
        img_fdg = nib.load(img_fdg).get_fdata()

        img_mri = norm(img_mri)
        img_fdg = norm(img_fdg)

        img_mri = nib.Nifti1Image(img_mri, np.eye(4))
        nib.save(img_mri, os.path.join(adni1_path, subj, f'{subj}_MRI_mask_norm_crop_fsl.nii.gz'))
        img_fdg = nib.Nifti1Image(img_fdg, np.eye(4))
        nib.save(img_fdg, os.path.join(adni1_path, subj, f'{subj}_fdg_mask_norm_crop_fsl.nii.gz'))


# with multiprocessing.Pool(10) as p:
#     print("resize,crop, augmentation image")
#     print("resize,crop, augmentation image")
#     print("resize,crop, augmentation image")
#     p.map(run_resize, subj_id)
#
# with multiprocessing.Pool(10) as p:
#     print("Crop image")
#     print("Crop image")
#     print("Crop image")
#     p.map(run_crop, subj_id)

# subj_id = os.listdir('/home/icml/Downloads/adni2testmri/Data_ADNI2_ANTS')
subj_id = list(pd.read_csv('/home/icml/Downloads/label_ADNI3_60d.csv')['participant_id'])
subj_id.sort()
with multiprocessing.Pool(12) as p:
    print('ADNI3 Crop Image')
    p.map(run_crop_adni3, subj_id)
# subj_id = os.listdir('/home/id202188508/all/Data_ADNI1_ANTS')
# subj_id.sort()
# with multiprocessing.Pool(20) as p:
#     print("ADNI1 Crop Image")
#     p.map(run_crop_adni1, subj_id)

print("FINISH")
