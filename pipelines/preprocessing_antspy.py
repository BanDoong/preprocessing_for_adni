import os
import ants
import nibabel as nib
from nilearn.image import resample_to_img
from tqdm import tqdm


class Preprocessing_ANTs(object):
    """
    1) Bias correction with N4 algorithm from ANTs.
    2) Linear registration to template with RegistrationSynQuick from ANTs.
    3) Crop the background (in order to save computational power).
    Saving Preprocessed Image
    """

    def __init__(self, template, nii_directory, modality, ref_crop, save_path):
        self.nii_directory = nii_directory
        self.template_path = template
        self.ref_crop_path = ref_crop
        self.template = self.read_img(template)
        self.modality = modality
        self.ref_crop = self.read_img(ref_crop)
        self.subj_list = os.listdir(nii_directory)
        self.save_path = save_path
        self.subj_list.sort()
        for subj in tqdm(self.subj_list):
            subjdir_path = os.path.join(nii_directory, subj)
            mri, av45, av1451 = self.get_subdir_and_modality(subjdir_path)
            if modality == 't1':
                print(f'Preprocessing the data : {mri}')
                self.preprocessing_t1(mri, subj)
            elif modality == 'pet':
                # mri = os.path.join(self.save_path, subj, f'{subj}_MRI.nii.gz')
                print(f'Preprocessing the data : {av45}')
                out_pet = os.path.join(self.save_path, subj, f'{subj}_Amyloid.nii.gz')
                self.preprocessing_pet(mri, av45, out_pet)
                print(f'Preprocessing the data : {av1451}')
                out_pet = os.path.join(self.save_path, subj, f'{subj}_Tau.nii.gz')
                self.preprocessing_pet(mri, av1451, out_pet)
            else:
                raise Exception("You have only choice one of the 't1' or 'pet or other modality'")

            # del tmp data
            os.system('rm /tmp/tmp*.nii.gz')
            os.system('rm /tmp/tmp*.mat')

    def preprocessing_t1(self, img_path, subj):
        t1 = self.read_img(img_path)
        t1 = self.bias_correction(t1)
        # need to add out_path
        _ = self.registration_for_t1(fix=self.template, mov=t1,
                                     out=os.path.join(self.save_path, subj, f'{subj}_MRI.nii.gz'))
        output_img = self.crop_img(os.path.join(self.save_path, subj, f'{subj}_MRI.nii.gz'))
        nib.save(output_img, os.path.join(self.save_path, subj, f'{subj}_MRI_cropped.nii.gz'))

    def preprocessing_pet(self, t1, img_path, out_pet):
        """
        PET data must be preprocessed data in ADNI
        """
        # Registration to T1 (in MNI space)
        t1 = self.read_img(t1)
        self.registration_for_pet(fix=self.template, mov=t1, in_pet=img_path, out_pet=out_pet)
        # Cropping
        output_img = self.crop_img(out_pet)
        # Saving
        nib.save(output_img, out_pet[:-7] + '_cropped.nii.gz')

    def get_subdir_and_modality(self, subjdir_path):
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

    def read_img(self, img_path):
        img = ants.image_read(img_path)
        return img

    def bias_correction(self, img):
        img = ants.n4_bias_field_correction(img, spline_param=600)
        return img

    def crop_img(self, input_img):
        """Crop input image based on the reference.
        Args:
            input_img (str): image path to be processed
        """
        crop_img = resample_to_img(input_img, self.ref_crop_path, force_resample=True, interpolation='nearest')
        return crop_img

    def registration_for_t1(self, fix, mov, out):
        reg_output = ants.registration(fix, mov, type_of_transforme='TRSAA')  # matrix
        ants.image_write(reg_output['warpedmovout'], out)  # out : t1 output
        return reg_output
        # warpedmovout
        # antsRegistrationSyNQuick

    def registration_for_pet(self, fix, mov, in_pet, out_pet):
        in_pet = self.read_img(in_pet)
        t1_to_temp = self.registration_for_t1(fix=fix, mov=mov, out=out_pet)

        pet_to_t1 = ants.registration(mov, in_pet, type_of_transforme='TRSAA')  # matrix

        # ants_registration_node.inputs.fixed_image = self.ref_template
        # ants_registration_node.inputs.transform_type = "a"
        # ants_registration_node.inputs.dimension = 3

        out_in_pet = ants.apply_transforms(fixed=self.template, moving=pet_to_t1['warpedmovout'],
                                           transformlist=t1_to_temp['fwdtransforms'])
        print(out_in_pet)
        print(out_pet)
        ants.image_write(out_in_pet, out_pet)

        # print('***********', fix, mov, os.path.join(self.save_path, subj), '************')

###############################################################################
# template, nii_directory, modality, path, args, fix_img, ref_crop, t1, pet, save_path
atlas_path = '/home/icml/Downloads/tpl-MNI152NLin2009cSym'
template = os.path.join(atlas_path, 'mni_icbm152_t1_tal_nlin_sym_09c.nii')
ref_crop = os.path.join(atlas_path, 'ref_cropped_template.nii.gz')
nii_diretory = '/home/icml/Desktop/test'
modality = 't1'
save_path = '/home/icml/Desktop/test_2'

Preprocessing_ANTs(template=template, nii_directory=nii_diretory, modality=modality, ref_crop=ref_crop,
                   save_path=save_path)
modality = 'pet'
Preprocessing_ANTs(template=template, nii_directory=nii_diretory, modality=modality, ref_crop=ref_crop,
                   save_path=save_path)
###############################################################################