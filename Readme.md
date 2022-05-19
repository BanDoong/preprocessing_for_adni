This Guide based on the ubuntu environment.
You have to have below Programs to use pipelines

1) ants
2) spm, matlab


Installation Guide
- 
- Require Python library
  - numpy==1.21.2, antspyx==0.3.2, nipype==1.8.1, nibabel==2.5.1, nilearn==0.7.1
- Ants
   1) ants can download this site https://github.com/stnava/ANTs/releases
   2) follow the instruction https://github.com/ANTsX/ANTs/wiki/Compiling-ANTs-on-Linux-and-Mac-OS
   3) set the path in your ".zshrc" or ".bashrc" like this
      - export ANTSPATH=/home/icml/install/bin 
      - export PATH=${ANTSPATH}:${PATH}

- SPM12
   1) Download and the instruction are here
      - https://en.wikibooks.org/wiki/SPM/Installation_on_64bit_Linux
   2) Set the path in your ".zshrc" or ".bashrc" like this
      - export SPM_HOME='/home/icml/Downloads/spm12'

- Matlab
  1) Download and install the matlab
  2) Set the path in your ".zshrc" or ".bashrc" like this
     - export MATLAB_HOME='/home/icml/Downloads/matlab_R2020b_glnxa64/bin'
     - export PATH=${MATLAB_HOME}:$PATH
     - export MATLABCMD="${MATLAB_HOME}/matlab"
  3) Set the path spm12 in matlab by 
     1) [Set Path]
     2) [Add Folder...]
     3) Choose your spm12 folder
     4) [save]
     5) [click]


Preprocessing
-

1) ANTS
   1) Set the variable
      - atlas_path = " Altas path folder "
      - template = " Template nifti file path "
      - ref_crop = " Cropped Template nifti file path "
      - subj_path = " Subjects directory path "
      - save_path = " Output directory "
   2) python preprocessing_ants.py or python preprocessing_antspy.py
      - preprocessing_ants.py : nipype version ants
      - preprocessing_antspy.py : python version ants
2) SPM12
    1) Set the variable
       - subj_path = " Raw Data folder Path "
       - save_path = " Save folder Path"
    2) python preprocessing_spm.py