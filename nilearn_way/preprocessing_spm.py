import os
from nipype.interfaces import spm
import numpy as np
from nipype.interfaces.base import (
    File,
    InputMultiPath,
    OutputMultiPath,
    TraitedSpec,
    traits,
)
from nipype.interfaces.spm.base import SPMCommand, SPMCommandInputSpec
from nipype.utils.filemanip import filename_to_list, list_to_filename
from tqdm import tqdm


class ApplySegmentationDeformationInput(SPMCommandInputSpec):
    deformation_field = File(
        exists=True,
        mandatory=True,
        field="comp{1}.def",
        desc="SPM Segmentation deformation file",
    )
    in_files = InputMultiPath(
        File(exists=True),
        mandatory=True,
        field="out{1}.pull.fnames",
        desc="Files on which deformation field is applied",
    )
    interpolation = traits.Range(
        low=0,
        high=7,
        field="out{1}.pull.interp",
        desc="degree of b-spline used for interpolation",
    )
    mask = traits.Int(
        0, usedefault=True, field="out{1}.pull.mask", desc="image masking"
    )
    fwhm = traits.List(
        traits.Float(0),
        field="out{1}.pull.fwhm",
        minlen=3,
        maxlen=3,
        desc="3-element list (opt)",
    )


class ApplySegmentationDeformationOutput(TraitedSpec):
    out_files = OutputMultiPath(File(exists=True), desc="Transformed files")


class ApplySegmentationDeformation(SPMCommand):
    """Uses SPM to apply a deformation field obtained from Segmentation routine to a given file
    Examples
    --------
    # >>> import clinica.pipelines.t1_volume_tissue_segmentation.t1_volume_tissue_segmentation_utils as seg_utils
    # >>> inv = seg_utils.ApplySegmentationDeformation()
    # >>> inv.inputs.in_files = 'T1w.nii'
    # >>> inv.inputs.deformation = 'y_T1w.nii'
    # >>> inv.run() # doctest: +SKIP
    """

    input_spec = ApplySegmentationDeformationInput
    output_spec = ApplySegmentationDeformationOutput

    _jobtype = "util"
    _jobname = "defs"

    def _format_arg(self, opt, spec, val):
        """Convert input to appropriate format for spm"""
        if opt == "deformation_field":
            return np.array([list_to_filename(val)], dtype=object)
        if opt == "in_files":
            return np.array(filename_to_list(val), dtype=object)
        return val

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_files"] = []
        for filename in self.inputs.in_files:
            _, fname = os.path.split(filename)
            outputs["out_files"].append(os.path.realpath("w%s" % fname))
        return outputs


def get_tissue_tuples(
        tissue_map,
        tissue_classes,
        dartel_tissues,
        save_warped_unmodulated,
        save_warped_modulated,
):
    """Get tissue tuples.
    Method to obtain the list of tuples, one for each tissue class, with the following fields:
     - tissue probability map (4D), 1-based index to frame
     - number of gaussians
     - which maps to save [Native, DARTEL] - a tuple of two boolean values
     - which maps to save [Unmodulated, Modulated] - a tuple of two boolean values
    Args:
        tissue_map: Path to tissue maps
        tissue_classes: Classes of images to obtain from segmentation. Ex: [1,2,3] is GM, WM and CSF
        dartel_tissues: Classes of images to save for DARTEL template calculation. Ex: [1] is only GM'
        save_warped_unmodulated: Save warped unmodulated images for tissues specified in --tissue_classes
        save_warped_modulated: Save warped modulated images for tissues specified in --tissue_classes
    Returns:
        List of tuples according to NewSegment input por tissues
    """
    tissues = []

    for i in range(1, 7):
        n_gaussians = 2

        if i == 4 or i == 5:
            n_gaussians = i - 1

        native_space = False
        dartel_input = False
        warped_unmodulated = False
        warped_modulated = False

        if i in tissue_classes:
            native_space = True
            if save_warped_unmodulated:
                warped_unmodulated = True
            if save_warped_modulated:
                warped_modulated = True

        if i in dartel_tissues:
            dartel_input = True

        tissues.append(
            (
                (tissue_map, i),
                n_gaussians,
                (native_space, dartel_input),
                (warped_unmodulated, warped_modulated),
            )
        )
    return tissues


def get_tpm():
    """Get Tissue Probability Map (TPM) from SPM.
    Returns:
        str: TPM.nii from SPM
    """
    import os
    from glob import glob
    from os.path import join

    spm_home = os.getenv("SPM_HOME")

    if not spm_home:
        # Try MCR to get a hint on SPM location
        spm_home = os.getenv("SPMSTANDALONE_HOME")

    if not spm_home:
        raise RuntimeError(
            "Could not determine location of your SPM installation. Neither $SPM_HOME "
            "or $SPMSTANDALONE_HOME are present in your environment"
        )

    tpm_file_glob = glob(join(spm_home, "**/TPM.nii"), recursive=True)
    if len(tpm_file_glob) == 0:
        raise RuntimeError(f"No file found for TPM.nii in your $SPM_HOME in {spm_home}")
    if len(tpm_file_glob) > 1:
        error_str = f"Multiple files found for TPM.nii in your SPM_HOME {spm_home}:"
        for file in tpm_file_glob:
            error_str += "\n\t" + file
        raise RuntimeError(error_str)
    return tpm_file_glob[0]


def segmentation(mov, save_path, subj, subj_folder, nifti):
    parameters = {}
    parameters.setdefault("tissue_classes", [1, 2, 3])
    parameters.setdefault("dartel_tissues", [1, 2, 3])
    parameters.setdefault("save_warped_unmodulated", True)
    parameters.setdefault("save_warped_modulated", False)
    parameters.setdefault("tissue_probability_maps", None)
    parameters["tissue_probability_maps"] = get_tpm()

    seg = spm.NewSegment()
    seg.inputs.channel_files = mov
    seg.inputs.channel_info = (0.0001, 60, (True, True))
    seg.inputs.write_deformation_fields = [True, True]
    seg.inputs.tissues = get_tissue_tuples(parameters["tissue_probability_maps"],
                                           parameters["tissue_classes"],
                                           parameters["dartel_tissues"],
                                           parameters["save_warped_unmodulated"],
                                           parameters["save_warped_modulated"])

    seg.run()

    os.system(
        f'mv {os.path.join(subj_folder, f"c1{nifti}")} {os.path.join(save_path, subj, f"{subj}_segm_graymatter.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"c2{nifti}")} {os.path.join(save_path, subj, f"{subj}_segm_whitematter.nii")}')
    os.system(f'mv {os.path.join(subj_folder, f"c3{nifti}")} {os.path.join(save_path, subj, f"{subj}_segm_csf.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"rc1{nifti}")} {os.path.join(save_path, subj, f"{subj}_dartel_input_segm_graymatter.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"rc2{nifti}")} {os.path.join(save_path, subj, f"{subj}_dartel_input_segm_whitematter.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"rc3{nifti}")} {os.path.join(save_path, subj, f"{subj}_dartel_input_segm_csf.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"wc1{nifti}")} {os.path.join(save_path, subj, f"{subj}_modulate_off_probability_segm_graymatter.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"wc2{nifti}")} {os.path.join(save_path, subj, f"{subj}_modulate_off_probability_segm_whitematter.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"wc3{nifti}")} {os.path.join(save_path, subj, f"{subj}_modulate_off_probability_segm_csf.nii")}')
    # os.system(
    #     f'mv {os.path.join(subj_folder, f"y_{nifti}")} {os.path.join(save_path, subj, f"{subj}_transform_forward_deformation.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"y_{nifti}")} {os.path.join(save_path, subj, f"y_{nifti}")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"iy_{nifti}")} {os.path.join(save_path, subj, f"{subj}_transform_inverse_deformation.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"m{nifti}")} {os.path.join(save_path, subj, f"{subj}_t1_biased_correction.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"BiasField_{nifti}")} {os.path.join(save_path, subj, f"{subj}_t1_bias_feild.nii")}')
    os.system(
        f'mv {os.path.join(subj_folder, f"{nifti[:-4]}_seg8.mat")} {os.path.join(save_path, subj, f"{subj}_seg8.mat")}')


def seg_to_deform(mov, deformation):
    t1_to_mni = ApplySegmentationDeformation()
    t1_to_mni.inputs.in_files = mov
    t1_to_mni.inputs.deformation_field = deformation
    t1_to_mni.run()


def get_subdir_and_modality(subjdir_path):
    subfile = os.listdir(subjdir_path)
    mri = None
    for file in subfile:
        if ('Accler' in file or 'MPRAGE' in file) and 'nii' in file:
            mri = os.path.join(subjdir_path, file)
            return mri, file
###############################################################################
python_file_path = '/media/icml/extremeSSD/all/preprocessing/pipelines'
subj_path = '/home/icml/Desktop/preprocessing_check/rawdata'
save_path = '/home/icml/Desktop/preprocessing_check/t1_volume_test'
subj_list = os.listdir(subj_path)
subj_list.sort()
###############################################################################
for subj in tqdm(subj_list):
    subj_folder = os.path.join(subj_path, subj)
    if not os.path.exists(os.path.join(save_path, subj)):
        os.makedirs(os.path.join(save_path, subj))
    mri, nifti = get_subdir_and_modality(os.path.join(subj_path, subj))
    segmentation(mri, save_path, subj, subj_folder, nifti)
    mov = os.path.join(save_path, subj, f'{subj}_t1_biased_correction.nii')
    # mov = os.path.join(mri)
    deformation = os.path.join(save_path, subj, f"y_{nifti}")
    seg_to_deform(mov, deformation)
    os.system(
        f'mv {os.path.join(save_path, subj, f"y_{nifti}")} {os.path.join(save_path, subj, f"{subj}_transform_forward_deformation.nii")}')
    os.system(f'mv {os.path.join(python_file_path,f"w{nifti}")} {os.path.join(save_path, subj, f"{subj}_space-Ixi549Space_T1w.nii")}')
    os.system(f'gzip -f {os.path.join(save_path, subj)}/*.nii')
    os.system(f'rm -f {os.path.join(save_path, subj)}/*.nii')

