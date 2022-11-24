from nipype.interfaces import ants
import os
from nilearn.image import resample_to_img
import nibabel as nib
from tqdm import tqdm

def crop_img(input_img=None, ref_crop_path=None):
    """Crop input image based on the reference.
    Args:
        input_img (str): image path to be processed
    """
    crop_img = resample_to_img(input_img, ref_crop_path, force_resample=True, interpolation='nearest')
    return crop_img


def n4_bias_correction_with_ants(moving_path, save_path, subj):
    n4 = ants.N4BiasFieldCorrection()
    # n4.inputs.output_transform_prefix = 'test'
    # n4.inputs.fixed_image = fix_path
    n4.inputs.input_image = moving_path
    n4.inputs.dimension = 3
    n4.inputs.save_bias = False
    n4.inputs.bspline_fitting_distance = 600
    n4.inputs.output_image = os.path.join(save_path, subj, f'{subj}_n4_mri.nii.gz')
    print(n4.cmdline)
    n4.run()


def registration(fix_path, moving_path, save_path, subj):
    from nipype.interfaces.ants import RegistrationSynQuick
    reg = RegistrationSynQuick()
    reg.inputs.fixed_image = fix_path
    reg.inputs.moving_image = moving_path
    reg.inputs.num_threads = 2
    reg.inputs.transform_type = "a"
    reg.inputs.dimension = 3
    reg.inputs.output_prefix = os.path.join(save_path, subj, f'{subj}_MRI')
    print(reg.cmdline)
    reg.run()


def get_subdir_and_modality(subjdir_path):
    subfile = os.listdir(subjdir_path)
    mri, av45, av1451 = None, None, None
    for file in subfile:
        if ('Accler' in file or 'MPRAGE' in file) and 'nii' in file:
            mri = os.path.join(subjdir_path, file)
        if ('AV45' in file or 'FBB' in file) and 'nii' in file:
            av45 = os.path.join(subjdir_path, file)
        if 'AV1451' in file and 'nii' in file:
            av1451 = os.path.join(subjdir_path, file)
    return mri, av45, av1451

###############################################################################
atlas_path = '/home/icml/Downloads/tpl-MNI152NLin2009cSym'
template = os.path.join(atlas_path, 'mni_icbm152_t1_tal_nlin_sym_09c.nii')
ref_crop = os.path.join(atlas_path, 'ref_cropped_template.nii.gz')
subj_path = '/home/icml/Desktop/test'
save_path = '/home/icml/Desktop/test_3'
###############################################################################

subj_list = os.listdir(subj_path)

for subj in tqdm(subj_list):
    if not os.path.exists(os.path.join(save_path, subj)):
        os.makedirs(os.path.join(save_path, subj))
    mri, _, _ = get_subdir_and_modality(os.path.join(subj_path, subj))
    n4_bias_correction_with_ants(moving_path=os.path.join(subj_path, mri), save_path=save_path, subj=subj)
    registration(fix_path=template, moving_path=os.path.join(save_path, subj, f'{subj}_n4_mri.nii.gz'),
                 save_path=save_path, subj=subj)
    os.system(f'mv {os.path.join(save_path, subj, f"{subj}_MRIWarped.nii.gz")} {os.path.join(save_path, subj, f"{subj}_MRI.nii.gz")}')
    cropped = crop_img(input_img=os.path.join(save_path, subj, f'{subj}_MRI.nii.gz'), ref_crop_path=ref_crop)
    nib.save(cropped, os.path.join(save_path, subj, f'{subj}_MRI_cropped.nii.gz'))

