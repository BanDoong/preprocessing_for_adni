import SimpleITK as sitk
import ants
import sys
import os
import nibabel as nib
import numpy as np

# maksOrNot = sys.argv[1]
# fix = ants.image_read(sys.argv[2])
# mov = ants.image_read(sys.argv[3])
# out = str(sys.argv[4])
# in_av45 = ants.image_read(sys.argv[5])
# out_av45 = str(sys.argv[6])
#
# in_av1451 = ants.image_read(sys.argv[5])
# out_av1451 = str(sys.argv[6])


"""
ORiginal Shape
import SimpleITK as sitk
import ants
import sys
import os
import nibabel as nib
import numpy as np

maksOrNot = sys.argv[1]
fix= ants.image_read(sys.argv[2])
mov= ants.image_read(sys.argv[3])
out= str(sys.argv[4])
pet = ants.image_read(sys.argv[5])
out_pet = str(sys.argv[6])

reg_output = ants.registration(fix, mov, type_of_transforme = 'SyN')
if maksOrNot=='mask':
    mask = ants.image_read(sys.argv[5])
    out_mask= ants.apply_transforms(fix, mask, transformlist=reg_output['fwdtransforms'])
    ants.image_write(out_mask, out)
else:
    ants.image_write(reg_output['warpedmovout'], out)
    out_pet = ants.apply_transforms(fix,pet,transformlist=reg_output['fwdtransforms'])
    ants.image_write(out_pet,out_pet)

print ('***********', maksOrNot, fix, mov, out,'************')


"""


def coreg_modality(maksOrNot=None, fix=None, mov=None, out=None, in_av45=None, out_av45=None, in_av1451=None,
                   out_av1451=None):
    fix = ants.image_read(fix)  # reference path
    mov = ants.image_read(mov)  # T1 path
    in_av45 = ants.image_read(in_av45)  # av45
    # in_av1451 = ants.image_read(in_av1451)  # av1451
    reg_output = ants.registration(fix, mov, type_of_transforme='SyN')  # matrix
    # if maksOrNot == 'mask':
    #     # mask = ants.image_read(sys.argv[5])
    #     # out_mask = ants.apply_transforms(fix, mask, transformlist=reg_output['fwdtransforms'])
    #     # ants.image_write(out_mask, out)
    #     pass
    # else:
    ants.image_write(reg_output['warpedmovout'], out)  # out : t1 output
    out_in_av45 = ants.apply_transforms(fixed=fix, moving=in_av45, transformlist=reg_output['fwdtransforms'])
    ants.image_write(out_in_av45, out_av45)
    # out_in_av1451 = ants.apply_transforms(fixed=fix, moving=in_av1451, transformlist=reg_output['fwdtransforms'])
    # ants.image_write(out_in_av1451, out_av1451)

    print('***********', out, out_av45, '************')
