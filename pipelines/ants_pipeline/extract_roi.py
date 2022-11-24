# import ants
import os
import numpy as np
import nibabel as nib
import argparse
import multiprocessing
# import multiprocessing

parser = argparse.ArgumentParser("EXTRACTING ROI")
parser.add_argument('--path', default='/home/id202188508/all/freesurfer_RAS/adni3/free7_RAS/')

args = parser.parse_args()


def crop_(mask_img, img, patch_size):
    x_h = patch_size[0] // 2
    y_h = patch_size[1] // 2
    z_h = patch_size[2] // 2

    non_zero = np.nonzero(mask_img)

    x_min = non_zero[0].min()
    x_max = non_zero[0].max()
    y_min = non_zero[1].min()
    y_max = non_zero[1].max()
    z_min = non_zero[2].min()
    z_max = non_zero[2].max()
    x_mid = int(np.round(np.median([x_min, x_max])))
    y_mid = int(np.round(np.median([y_min, y_max])))
    z_mid = int(np.round(np.median([z_min, z_max])))
    if x_mid - x_h < 0:
        x_mid = x_h
    elif y_mid - y_h < 0:
        y_mid = y_h
    elif z_mid - z_h < 0:
        z_mid = z_mid

    # cropped = img[x_mid - x_h:x_mid + x_h, y_mid - y_h:y_mid + y_h, z_mid - z_h:z_mid + z_h]
    cropped = img[x_mid - x_h:x_mid + x_h, :, :]
    cropped = cropped[:, y_mid - y_h:y_mid + y_h, :]
    cropped = cropped[:, :, z_mid - z_h:z_mid + z_h]

    return cropped


def z_score_norm(img_1):
    img_1 = (img_1 - np.mean(img_1)) / np.std(img_1)
    return img_1


def min_max_norm(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


if 'adni3' in args.path or 'free7_RAS' in args.path:
    data_folder = '/home/icml/Desktop/Data_ANTS_PVC'
    modality = ['MRI', 'Tau', 'Amyloid']
    ban_list = ['020_S_6227', '036_S_4715']
    subj_list_1_padding_hippo = ['007_S_6515', '007_S_6521', '016_S_6839', '020_S_6449', '027_S_4926', '027_S_6034',
                                 '027_S_6183', '036_S_6179', '041_S_4513', '041_S_4876', '041_S_6159', '057_S_6869',
                                 '082_S_4428', '082_S_6564', '094_S_4630', '094_S_6468', '099_S_6691', '114_S_6462',
                                 '114_S_6813', '126_S_6721', '127_S_4301', '128_S_4586', '135_S_6110', '135_S_6389',
                                 '153_S_6237', '168_S_6828']
    subj_list_2_padding_hippo = ['007_S_2394', '014_S_6145', '024_S_6385', '100_S_6578', '168_S_6828', '094_S_6468',
                                 '135_S_6389']
    subj_list_1_padding_temporal = ['016_S_6381', '019_S_6315', '020_S_6227', '035_S_6722', '036_S_4491', '053_S_4813',
                                    '116_S_6133', '135_S_6586', '141_S_6015']
    subj_list_2_padding_temporal = ['014_S_4576']
    subj_list_3_padding_hippo = ['033_S_6266', '068_S_4332']
    subj_list_3_padding_temporal = ['033_S_6266', '094_S_6250', '168_S_6321']
elif 'adni1' in args.path:
    data_folder = '/home/id202188508/all/Data_ADNI1_ANTS'
    modality = ['MRI', 'fdg']
    ban_list = ['100_S_0747', '003_S_1257', '031_S_0830']
    subj_list_1_padding_hippo = ['002_S_0729', '013_S_0325', '016_S_0991', '023_S_1046', '029_S_0843', '035_S_0341',
                                 '035_S_0555', '941_S_1295']
    subj_list_2_padding_hippo = ['002_S_1261', '052_S_0989', '128_S_0545']
    subj_list_1_padding_temporal = ['062_S_1099', '114_S_1118', '052_S_1352']
    subj_list_2_padding_temporal = ['012_S_0932', '052_S_0989']

    subj_list_3_padding_hippo = []
    subj_list_3_padding_temporal = []
else:
    data_folder = None
    modality = None
    ban_list = []
    subj_list_1_padding_hippo = []
    subj_list_2_padding_hippo = []
    subj_list_1_padding_temporal = []
    subj_list_2_padding_temporal = []
    subj_list_3_padding_hippo = []
    subj_list_3_padding_temporal = []

option = ['hippo', 'hippo_temporal']

subj_list = os.listdir(data_folder)
subj_list.sort()

# for subj in subj_list:
def processing(subj):
    if subj in ban_list:
        print(f'{subj} is passed')
    else:
        print(f'{subj} is preprocessing')
        img_category = ['_left', '_right']
        reg_path = os.path.join(args.path, subj, f'reg_label/{subj}_reg.nii.gz')
        output_folder = '/home/icml/Desktop/Data_ANTS_PVC'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, subj)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        # Hippocampus
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_hippo.nii.gz --match 17 53 > /dev/null')
        os.system(
            f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_hippo_right.nii.gz --match 53 > /dev/null')
        os.system(
            f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_hippo_left.nii.gz --match 17 > /dev/null')
        #
        # # inferior temporal
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_inferior.nii.gz --match 1009 2009 > /dev/null')
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_inferior_right.nii.gz --match 2009 > /dev/null')
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_inferior_left.nii.gz --match 1009 > /dev/null')
        #
        # # middle temporal
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_middle.nii.gz --match 1015 2015 > /dev/null')
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_middle_right.nii.gz --match 2015 > /dev/null')
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_middle_left.nii.gz --match 1015 > /dev/null')
        #
        # # Hippocampus, inferior temporal, middle temporal
        # os.system(
        #     f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_hippo_temporal.nii.gz --match 17 1009 1015 53 2009 2015 > /dev/null')
        os.system(
            f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_hippo_temporal_left.nii.gz --match 17 1009 1015 > /dev/null')
        os.system(
            f'mri_binarize --i {reg_path} --o {output_path}/{subj}_mask_hippo_temporal_right.nii.gz --match 53 2009 2015 > /dev/null')


        for region in option:
            left_label_path = os.path.join(data_folder, subj) + '/' + subj + f'_mask_{region}_left.nii.gz'
            left_label_img = nib.load(left_label_path).get_fdata()[91:, :, :]
            right_label_path = os.path.join(data_folder, subj) + '/' + subj + f'_mask_{region}_right.nii.gz'
            right_label_img = nib.load(right_label_path).get_fdata()[:91, :, :]
            if region == 'hippo':
                patch_size = [50, 50, 50]
            else:
                patch_size = [80, 96, 80]
            for m in modality:
                img_path = os.path.join(data_folder, subj, f'{subj}_{m}_mask.nii.gz')
                img_orig = nib.load(img_path).get_fdata()

                img_left = crop_(left_label_img, img_orig[91:, :, :], patch_size)
                img_right = crop_(right_label_img, img_orig[:91, :, :], patch_size)

                if subj in subj_list_1_padding_temporal or subj in subj_list_1_padding_hippo:
                    if region == 'hippo':
                        if np.shape(img_left) == (49, 50, 50):
                            img_left = np.pad(img_left, [(1, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                        if np.shape(img_right) == (49, 50, 50):
                            img_right = np.pad(img_right, [(1, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                    else:
                        if np.shape(img_left) == (79, 96, 80):
                            img_left = np.pad(img_left, [(1, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                        if np.shape(img_right) == (79, 96, 80):
                            img_right = np.pad(img_right, [(1, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)

                if subj in subj_list_2_padding_temporal or subj in subj_list_2_padding_hippo:
                    if region == 'hippo':
                        if np.shape(img_left) == (48, 50, 50):
                            img_left = np.pad(img_left, [(2, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                        if np.shape(img_right) == (48, 50, 50):
                            img_right = np.pad(img_right, [(2, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                    else:
                        if np.shape(img_left) == (78, 96, 80):
                            img_left = np.pad(img_left, [(2, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                        if np.shape(img_right) == (78, 96, 80):
                            img_right = np.pad(img_right, [(2, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                if subj in subj_list_3_padding_temporal or subj in subj_list_3_padding_hippo:
                    if region == 'hippo':
                        if np.shape(img_left) == (47, 50, 50):
                            img_left = np.pad(img_left, [(3, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                        if np.shape(img_right) == (47, 50, 50):
                            img_right = np.pad(img_right, [(3, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                    else:
                        if np.shape(img_left) == (77, 96, 80):
                            img_left = np.pad(img_left, [(3, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)
                        if np.shape(img_right) == (77, 96, 80):
                            img_right = np.pad(img_right, [(3, 0), (0, 0), (0, 0)], mode='constant', constant_values=0)

                # z_norm_img_left = z_score_norm(img_left)
                # z_norm_img_left = nib.Nifti1Image(z_norm_img_left, np.eye(4))
                # nib.save(z_norm_img_left, os.path.join(data_folder, subj, f'{subj}_{m}_znorm_{region}_left.nii.gz'))
                #
                # z_norm_img_right = z_score_norm(img_right)
                # z_norm_img_right = nib.Nifti1Image(z_norm_img_right, np.eye(4))
                # nib.save(z_norm_img_right, os.path.join(data_folder, subj, f'{subj}_{m}_znorm_{region}_right.nii.gz'))

                min_max_img_left = min_max_norm(img_left)
                min_max_img_left = nib.Nifti1Image(min_max_img_left, np.eye(4))
                nib.save(min_max_img_left, os.path.join(data_folder, subj, f'{subj}_{m}_maxnorm_{region}_left.nii.gz'))

                min_max_img_right = min_max_norm(img_right)
                min_max_img_right = nib.Nifti1Image(min_max_img_right, np.eye(4))
                nib.save(min_max_img_right,
                         os.path.join(data_folder, subj, f'{subj}_{m}_maxnorm_{region}_right.nii.gz'))

with multiprocessing.Pool(12) as p:
    p.map(processing, subj_list)
