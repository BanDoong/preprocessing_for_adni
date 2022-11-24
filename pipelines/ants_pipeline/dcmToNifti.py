import os
import argparse

parser = argparse.ArgumentParser(description='DCM to NIFTI')
parser.add_argument("--input", type=str, default='/home/icml/Downloads/adni2testmri/dcm_av45')
parser.add_argument("--output", type=str, default='/home/icml/Downloads/adni2testmri/nifti_av45')
args = parser.parse_args()

path = args.input
output = args.output
if not os.path.exists(output):
    os.makedirs(output)

subj_list = os.listdir(path)
subj_list.sort()
for subj in subj_list:
    total_path = os.path.join(path, subj)
    output_path = os.path.join(output, subj)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    os.system(f'dcm2niix -o {output_path} {total_path}')
