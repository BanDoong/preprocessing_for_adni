# filter(lambda x: x.requires_grad, model.parameters()
# from run_finetune import add_element
import os
from torchvision.models import resnet18
import torch.nn as nn
from model_utils import PadMaxPool3d, Flatten
import numpy as np
import pandas as pd
from datetime import datetime

# path = '/final_labels/getlabels'
# train_0 = pd.read_csv(os.path.join(path, 'train_splits-5', 'split-0', 'AD.tsv'), sep='\t')
# train_0 = pd.concat([train_0, pd.read_csv(os.path.join(path, 'train_splits-5', 'split-0', 'CN.tsv'), sep='\t')],
#                     ignore_index=True)
# train_0 = pd.concat([train_0, pd.read_csv(os.path.join(path, 'train_splits-5', 'split-0', 'MCI.tsv'), sep='\t')],
#                     ignore_index=True)
# train_0 = pd.concat(
#     [train_0, pd.read_csv(os.path.join(path, 'validation_splits-5', 'split-0', 'CN_baseline.tsv'), sep='\t')],
#     ignore_index=True)
# train_0 = pd.concat(
#     [train_0, pd.read_csv(os.path.join(path, 'validation_splits-5', 'split-0', 'MCI_baseline.tsv'), sep='\t')],
#     ignore_index=True)
# train_0 = pd.concat(
#     [train_0, pd.read_csv(os.path.join(path, 'validation_splits-5', 'split-0', 'AD_baseline.tsv'), sep='\t')],
#     ignore_index=True)
#
# # train_0.to_csv('/home/icml/Downloads/test.csv')
# subj_list = list(train_0['participant_id'])
# tmp = list(train_0['session_id'])
# final_list = []
# session_list = []
# for k in range(len(subj_list)):
#     final_list.append((subj_list[k][8:11] + '_S_' + subj_list[k][12:16]))
#     if tmp[k] == 'ses-M00':
#         session_list.append('bl')
#     elif len(tmp[k]) == 7:
#         session = 'm' + tmp[k][-2] + tmp[k][-1]
#         session_list.append(session)
#     else:
#         session = 'm' + tmp[k][-3] + tmp[k][-2] + tmp[k][-1]
#         session_list.append(session)
#
#
# # print(final_list)
# def unique(list1):
#     # initialize a null list
#     unique_list = []
#
#     # traverse for all elements
#     for x in list1:
#         # check if exists in unique_list or not
#         if x not in unique_list:
#             unique_list.append(x)
#     return unique_list
#     # print list
#     # for x in unique_list:
#     #
#
#
# # final_list = final_list
# print(len(final_list))
# # print(session_list)
# print(len(session_list))
# save_path = '/home/icml/Downloads/ADNIMERGE 21.xlsx'
# df = pd.read_excel(save_path)
# df.to_csv('/home/icml/Downloads/ADNIMERGE.csv')

"""
RID is different each ADNI season
"""


def check_stability(subj_df, period_session):
    baseline_diganosis = subj_df['DX_bl']
    if baseline_diganosis == 'SMC':
        baseline_diganosis = 'CN'
    elif baseline_diganosis == 'LMCI' or baseline_diganosis == 'EMCI':
        baseline_diganosis = 'MCI'
    period_diagnosis = subj_df[subj_df['VISCODE'] == period_session]['DX']
    if period_diagnosis == 'Dementia':
        period_diagnosis = 'AD'
    if baseline_diganosis == period_diagnosis:
        return baseline_diganosis


def neighbour_session(session, session_list, neighbour):
    temp_list = session_list + [session]
    temp_list.sort()
    print(temp_list)
    index_session = temp_list.index(session)
    if (index_session + neighbour) < 0 or index_session + neighbour >= len(temp_list):
        return None
    else:
        if int(temp_list[index_session + neighbour]) < 10:
            return '0' + str(temp_list[index_session + neighbour])
        else:
            return str(temp_list[index_session + neighbour])


def make_sMCI_pMCI(subj_df, n_months, current_series):
    subj_df = subj_df.sort_values(by='VISCODE')
    subj_df = subj_df.dropna(subset=['DX'])
    if subj_df.empty:
        print(f'Subject has no DX data')
        return 'not_stable', False
    tmp = list(subj_df['VISCODE'])


    tmp.sort()

    # print(tmp)
    # print(subj_df)
    if tmp[0] == 'bl':
        tmp[0] = '00'
    for t in range(len(tmp)):
        if t == 0 and tmp[t] == 'bl':
            tmp[t] = '00'
        else:
            tmp[t] = tmp[t].replace('m', '')
    tmp.sort(key=int)


    stable = False
    output = False
    i = 0
    time_step = str(int(tmp[0]) + n_months)

    # if time_step not in tmp:
    #     time_step = neighbour_session(time_step, tmp, +1)
    #     print(time_step)
    #
    # for r in range(len(tmp)):
    #     if tmp[r] == time_step:
    #         tmp = tmp[:r + 1]
    #         break

    output = []
    stable = []

    # if len(tmp) == 1 and not current_series == 'ADNI3':
    # if len(tmp) == 1:
    #     print(f'Subject : {list(subj_df["PTID"])[0]} has only baseline data or 1 Data range')
    #     return 'not_stable', False

    for viscode in tmp:
        if i == 0:
            if viscode == '00':
                diagnosis_bl = list(subj_df[subj_df['VISCODE'] == 'bl']['DX'])[0]
            else:
                diagnosis_bl = list(subj_df[subj_df['VISCODE'] == 'm' + viscode]['DX'])[0]
        else:
            diagnosis = list(subj_df[subj_df['VISCODE'] == 'm' + viscode]['DX'])[0]
            if diagnosis != diagnosis_bl:
                if diagnosis_bl == 'MCI' and diagnosis == 'Dementia':
                    output.append('pMCI')
                    stable.append('False')
                elif diagnosis_bl == 'CN' and diagnosis == 'MCI':
                    output.append('not_stable')
                    stable.append('False')
                elif diagnosis_bl == 'CN' and diagnosis == 'Dementia':
                    output.append('not_stable')
                    stable.append('False')
                elif diagnosis_bl == 'Dementia' and diagnosis == 'MCI':
                    output.append('not_stable')
                    stable.append('False')
                elif diagnosis_bl == 'Dementia' and diagnosis == 'CN':
                    output.append('not_stable')
                    stable.append('False')
            else:
                if diagnosis_bl == 'MCI' and diagnosis == 'MCI':
                    output.append('sMCI')
                    stable.append('True')
                elif diagnosis_bl == 'CN' and diagnosis == 'CN':
                    output.append('CN')
                    stable.append('True')
                elif diagnosis_bl == 'Dementia' and diagnosis == 'Dementia':
                    output.append('Dementia')
                    stable.append('True')
        i += 1
    # print(list(subj_df['PTID'])[0], output, stable)
    # print(list(subj_df['PTID'])[0], output, stable)
    if 'False' not in stable:
        if tmp[0] == '00':
            output = list(subj_df[subj_df['VISCODE'] == 'bl']['DX'])[0]
        else:
            output = list(subj_df[subj_df['VISCODE'] == 'm' + tmp[0]]['DX'])[0]
        stable = True
        if output == 'MCI':
            output = 'sMCI'
    elif 'pMCI' in output:
        output = 'pMCI'
    elif 'not_stable' in output:
        print(f'Subject : {list(subj_df["PTID"])[0]} is not stable')
        # print(list(subj_df['PTID'])[0], output, stable)
        output = 'not_stable'
        stable = False
    return output, stable


def removing_from_MRI_PET(final_list, session_list, adni_path, mri_path, pet_path, get_subject_id, current_series,
                          time_step, n_months):
    # mri_path = '/home/icml/Downloads/ADNI_DATA/ADNIMERGE.csv'
    # pet_path = '/home/icml/Downloads/ADNI_DATA/TAUMETA3.csv'
    # amy_path = '/home/icml/Downloads/ADNI_DATA/AMYMETA.csv'
    # get_subject_id = '/home/icml/Downloads/ADNI_DATA/ROSTER.csv'
    adni_df = pd.read_csv(adni_path)
    mri_df = pd.read_excel(mri_path)
    subj_id_df = pd.read_csv(get_subject_id)
    # adni3_different_list = ['016_S_5057', '007_S_4488', '007_S_4620', '007_S_4637', '012_S_4643', '013_S_4580',
    #                         '014_S_4401', '014_S_4576', '019_S_4367', '029_S_4585', '031_S_4021', '033_S_4176',
    #                         '041_S_4427',
    #                         '068_S_4424', '070_S_4856', '082_S_4224', '094_S_4649', '098_S_4003', '099_S_4076',
    #                         '100_S_4469',
    #                         '127_S_4604', '129_S_4369', '130_S_4343', '141_S_0767', '941_S_4292', '941_S_4376',
    #                         '002_S_1155',
    #                         '003_S_1122', '023_S_4115', '027_S_4869', '041_S_4510', '068_S_4431', '082_S_2121',
    #                         '128_S_2220',
    #                         '130_S_4417', '137_S_4536', '141_S_1052', '141_S_2333', '941_S_4036', '018_S_4399',
    #                         '021_S_4335',
    #                         '033_S_4177', '116_S_4043', '135_S_4598', '022_S_5004', '135_S_4489', '141_S_4160',
    #                         '022_S_5004',
    #                         '135_S_4489', '141_S_4160']
    adni3_adding_list = ['002_S_4262', '002_S_4521', '007_S_2394', '018_S_0142', '018_S_4809', '018_S_4889',
                         '019_S_4548', '027_S_2183', '027_S_2336', '027_S_4926', '027_S_5127', '032_S_2247',
                         '035_S_4414', '037_S_4028', '037_S_4071', '037_S_4302', '037_S_5126', '037_S_5222',
                         '053_S_5202', '073_S_4552', '094_S_2367', '094_S_4162', '094_S_4234', '094_S_4630',
                         '094_S_4858', '109_S_2200', '109_S_4380', '127_S_0112', '127_S_0925', '127_S_4198',
                         '128_S_4586', '135_S_4281', '135_S_4309', '135_S_5269', '135_S_5273', '137_S_0668',
                         '137_S_0722', '137_S_0800', '137_S_1414', '137_S_4482', '137_S_4520', '137_S_4587']

    removed_diagnosis = pd.DataFrame(columns=['participant_id', 'session', 'diagnosis', 'sex', 'age', 'MMSE', 'CDR'])
    total_diagnosis = pd.DataFrame(columns=['participant_id', 'session', 'diagnosis', 'sex', 'age', 'MMSE', 'CDR'])
    baseline_diagnosis = pd.DataFrame(columns=['participant_id', 'session', 'diagnosis', 'sex', 'age', 'MMSE', 'CDR'])

    tmp = []
    out_subj = []
    final_list.sort()

    for i in range(len(final_list)):
        if final_list[i] == '100_S_0747':
            continue

        tmp_ptid = adni_df[adni_df['PTID'] == final_list[i]]
        tmp_ptid_not_current = tmp_ptid
        tmp_ptid = tmp_ptid[tmp_ptid['COLPROT'] == current_series]
        subj_rid = list(subj_id_df[subj_id_df['PTID'] == final_list[i]]['RID'])[0]

        if tmp_ptid.empty:
            tmp_df = pd.DataFrame({'participant_id': [final_list[i]], 'diagnosis': 'no_ADNIMERGE'})
            removed_diagnosis = pd.concat([removed_diagnosis, tmp_df], ignore_index=True)
        else:
            if final_list[i] in adni3_adding_list and current_series == 'ADNI3':
                tmp_ptid_not_current = tmp_ptid_not_current[tmp_ptid_not_current['COLPROT'] == 'ADNI2']
                tmp_session = list(tmp_ptid['EXAMDATE'])
                downloaded_data = list(mri_df[mri_df['Subject'] == final_list[i]]['Acq Date'])[0]
                downloaded_data = datetime.strptime(downloaded_data, '%m/%d/%Y')
                for t in range(len(tmp_session)):
                    tmp_session[t] = datetime.strptime(tmp_session[t], '%Y-%m-%d %H:%M:%S')
                    if abs(tmp_session[t] - downloaded_data).days > 60:
                        continue
                    else:
                        session_list[i] = tmp_ptid_not_current[tmp_ptid_not_current['EXAMDATE'] == tmp_session[t]][
                            'VISCODE']
            if current_series == "ADNI3":
                tmp_session = list(tmp_ptid['VISCODE'])
                tmp_session.sort()
                for t in range(len(tmp_session)):
                    if tmp_session[t] == 'bl':
                        tmp_session[t] = '00'
                    elif tmp_session[t] == 'y1':
                        tmp_session[t] = '12'
                    else:
                        tmp_session[t] = tmp_session[t].replace('m', '')
                tmp_session.sort(key=int)
                if 'bl' not in tmp_session:
                    session_list[i] = 'm' + tmp_session[0]
            else:
                tmp_ptid = tmp_ptid_current

            if session_list[i] == 'm00':
                tmp_adnimerge = tmp_ptid[tmp_ptid['VISCODE'] == 'bl']
            else:
                tmp_adnimerge = tmp_ptid[tmp_ptid['VISCODE'] == session_list[i]]
            output_diagnosis, stable = make_sMCI_pMCI(tmp_ptid, n_months, current_series)
            tmp_diag = tmp_adnimerge['DX']
            tmp_session = session_list[i]
            tmp_age = tmp_adnimerge['AGE']
            tmp_sex = tmp_adnimerge['PTGENDER']
            tmp_mmse = tmp_adnimerge['MMSE']
            tmp_cdr = tmp_adnimerge['CDRSB']
            tmp_rid = list(tmp_adnimerge['RID'])[0]

            ## check subj id ##
            if tmp_rid != subj_rid:
                raise Exception(f'wrong {tmp_rid} != {subj_rid}')

            subj_date = mri_df[mri_df['Subject'] == final_list[i]]
            mri_date = str(list(subj_date['MRIdate'])[0])
            if current_series == 'ADNI1':
                pet_date = str(list(subj_date['PETdate'])[0])
            elif 'TAU' in pet_path:
                pet_date = str(list(subj_date['AV1451date'])[0])
            elif 'AMY' in pet_path:
                pet_date = str(list(subj_date['AV45date'])[0])

            if pet_date == 'nan' or mri_date == 'nan':
                print(final_list[i])
                print(mri_date, pet_date)
                out_subj.append(final_list[i])
                # 'participant_id', 'session', 'diagnosis', 'sex', 'age', 'MMSE', 'CDR'
                tmp_df = pd.DataFrame(
                    {'participant_id': [final_list[i]], 'session': tmp_session, 'diagnosis': tmp_diag,
                     'sex': tmp_sex, 'age': tmp_age, 'MMSE': tmp_mmse, 'CDR': tmp_cdr})
                removed_diagnosis = pd.concat([removed_diagnosis, tmp_df], ignore_index=True)
            else:
                # if current_series == 'ADNI1':
                mri_date = datetime.strptime(mri_date, '%m/%d/%Y')
                pet_date = datetime.strptime(pet_date, '%m/%d/%Y')
                # else:
                #     mri_date = datetime.strptime(mri_date, '%Y-%m-%d')
                #     pet_date = datetime.strptime(pet_date, '%Y-%m-%d')
                days_delta = abs((mri_date - pet_date).days)

                if days_delta > time_step:
                    out_subj.append(final_list[i])
                    # tmp_df = pd.DataFrame({'participant_id': [final_list[i]], 'diagnosis': tmp_diag})
                    tmp_df = pd.DataFrame(
                        {'participant_id': [final_list[i]], 'session': tmp_session, 'diagnosis': tmp_diag,
                         'sex': tmp_sex, 'age': tmp_age, 'MMSE': tmp_mmse, 'CDR': tmp_cdr})
                    removed_diagnosis = pd.concat([removed_diagnosis, tmp_df], ignore_index=True)
                else:
                    # tmp_df = pd.DataFrame({'participant_id': [final_list[i]], 'diagnosis': output_diagnosis})
                    tmp_df = pd.DataFrame(
                        {'participant_id': [final_list[i]], 'session': tmp_session, 'diagnosis': output_diagnosis,
                         'sex': tmp_sex, 'age': tmp_age, 'MMSE': tmp_mmse, 'CDR': tmp_cdr})
                    if output_diagnosis == 'not_stable':
                        removed_diagnosis = pd.concat([removed_diagnosis, tmp_df], ignore_index=True)
                    else:
                        total_diagnosis = pd.concat([total_diagnosis, tmp_df], ignore_index=True)
            tmp.append(final_list[i])
            if session_list[i] == 'm00':
                baseline_tmp = tmp_ptid[tmp_ptid['VISCODE'] == 'bl']
            else:
                baseline_tmp = tmp_ptid[tmp_ptid['VISCODE'] == session_list[i]]
            if baseline_tmp.empty:
                baseline_tmp = pd.DataFrame({'participant_id': [final_list[i]], 'diagnosis': 'nan'})
            else:
                baseline_tmp = pd.DataFrame(
                    {'participant_id': [final_list[i]], 'diagnosis': list(baseline_tmp['DX'])[0],
                     'session': tmp_session, 'sex': tmp_sex, 'age': tmp_age, 'MMSE': tmp_mmse, 'CDR': tmp_cdr})
            baseline_diagnosis = pd.concat([baseline_diagnosis, baseline_tmp], ignore_index=True)

    return baseline_diagnosis, total_diagnosis, removed_diagnosis


"""
########################ADNI3########################
mri_path = '/home/icml/Downloads/ADNI_DATA/ADNI3_Multi.xlsx'
adni_path = '/home/icml/Downloads/ADNI_DATA/ADNIMERGE.csv'
pet_path = '/home/icml/Downloads/ADNI_DATA/TAUMETA3.csv'
amy_path = '/home/icml/Downloads/ADNI_DATA/AMYMETA.csv'
get_subject_id = '/home/icml/Downloads/ADNI_DATA/ROSTER.csv'

time_step = 30
baseline_diagnosis_tau, total_tau, remove_tau = removing_from_MRI_PET(final_list, session_list, adni_path, mri_path,
                                                                      pet_path, get_subject_id, 'ADNI3', time_step, 36)
baseline_diagnosis_av45, total_av45, remove_av45 = removing_from_MRI_PET(final_list, session_list, adni_path, mri_path,
                                                                         amy_path, get_subject_id, 'ADNI3', time_step,
                                                                         36)
not_same_subj_av45 = []
remove_tau_list = list(remove_tau['PTID'])
remove_av45_list = list(remove_av45['PTID'])
remove_tau_list.sort()
remove_av45_list.sort()
for j in range(len(remove_av45_list)):
    if remove_av45_list[j] not in remove_tau_list:
        not_same_subj_av45.append(remove_av45_list[j])

# print(train_0['diagnosis'].value_counts())
test = remove_tau_list + not_same_subj_av45

print(baseline_diagnosis_tau['DX'].value_counts())

print(f'Total_removed_data : {len(test)}')
cn, mci, ad = 0, 0, 0
for t in test:
    t = 'sub-ADNI' + t[:3] + 'S' + t[6:]
    tmp = list(train_0[train_0['participant_id'] == t]['diagnosis'])[0]
    if tmp == 'AD':
        ad += 1
    elif tmp == 'MCI':
        mci += 1
    else:
        cn += 1
print(f'Removed No. Data  AD : {ad} , MCI : {mci}, CN : {cn}')
print('=' * 50)
print('=' * 50)

time_step = 60
baseline_diagnosis_tau, total_tau, remove_tau = removing_from_MRI_PET(final_list, session_list, adni_path, mri_path,
                                                                      pet_path, get_subject_id, 'ADNI3',
                                                                      time_step, 36)
baseline_diagnosis_av45, total_av45, remove_av45 = removing_from_MRI_PET(final_list, session_list, adni_path, mri_path,
                                                                         amy_path, get_subject_id, 'ADNI3',
                                                                         time_step, 36)
not_same_subj_av45 = []
remove_tau_list = list(remove_tau['PTID'])
remove_av45_list = list(remove_av45['PTID'])
remove_tau_list.sort()
remove_av45_list.sort()
for j in range(len(remove_av45_list)):
    if remove_av45_list[j] not in remove_tau_list:
        not_same_subj_av45.append(remove_av45_list[j])

# print(train_0['diagnosis'].value_counts())
test = remove_tau_list + not_same_subj_av45
print(f'Total_removed_data : {len(test)}')
cn, mci, ad = 0, 0, 0
for t in test:
    t = 'sub-ADNI' + t[:3] + 'S' + t[6:]
    tmp = list(train_0[train_0['participant_id'] == t]['diagnosis'])[0]
    if tmp == 'AD':
        ad += 1
    elif tmp == 'MCI':
        mci += 1
    else:
        cn += 1
print(f'Removed No. Data  AD : {ad} , MCI : {mci}, CN : {cn}')

####################################################
"""
"""
########################ADNI1########################
mri_path = '/home/icml/Downloads/ADNI_DATA/ADNI1_Multi.xlsx'
adni_path = '/home/icml/Downloads/ADNI_DATA/ADNIMERGE.csv'
pet_path = '/home/icml/Downloads/ADNI_DATA/PETMETA_ADNI1.csv'
get_subject_id = '/home/icml/Downloads/ADNI_DATA/ROSTER.csv'
subj_list = os.listdir('/home/icml/Downloads/ADNI1/Data_ADNI1_ANTS')
session_list = ['bl'] * len(subj_list)
time_step = 30
print(f'Number of Subjects {len(subj_list)}')
baseline_diagnosis, total, remove = removing_from_MRI_PET(subj_list, session_list, adni_path, mri_path, pet_path,
                                                          get_subject_id, 'ADNI1', time_step, 36)
print(f'Baselien DX')
print(baseline_diagnosis["diagnosis"].value_counts())
print(f'total DX')
print(total["diagnosis"].value_counts())
print(f'remove DX')
print(remove["diagnosis"].value_counts())

# t = list(total["PTID"])
# r = list(remove["PTID"])
# t.sort()
# r.sort()
# print(len(t), len(r))
# print(t)
# print(r)

time_step = 60
print('=' * 25, 'time_step : 60', '=' * 25)
print(f'Number of Subjects {len(subj_list)}')
baseline_diagnosis, total, remove = removing_from_MRI_PET(subj_list, session_list, adni_path, mri_path, pet_path,
                                                          get_subject_id, 'ADNI1', time_step, 36)
print(f'Baselien DX')
print(baseline_diagnosis["diagnosis"].value_counts())
print(f'total DX')
print(total["diagnosis"].value_counts())
print(f'remove DX')
print(remove["diagnosis"].value_counts())

baseline_diagnosis.to_csv('/home/icml/Downloads/ADNI1/total_labels.csv')
total.to_csv('/home/icml/Downloads/ADNI1/train_60d.csv')
remove.to_csv('/home/icml/Downloads/ADNI1/remove_60d.csv')

####################################################
print('=' * 50)
print('=' * 50)
"""

########################ADNI2########################



mri_path = '/home/icml/Downloads/ADNI_DATA/ADNI3_Multi_added_ADNI2.xlsx'
adni_path = '/home/icml/Downloads/ADNI_DATA/ADNIMERGE.csv'
pet_path = '/home/icml/Downloads/ADNI_DATA/TAUMETA3.csv'
pet_path_2 = '/home/icml/Downloads/ADNI_DATA/TAUMETA.csv'
amy_path = '/home/icml/Downloads/ADNI_DATA/AMYMETA.csv'
amy_path_2 = '/home/icml/Downloads/ADNI_DATA/AV45META.csv'
get_subject_id = '/home/icml/Downloads/ADNI_DATA/ROSTER.csv'
subj_list = os.listdir('/home/icml/Downloads/adni2testmri/concat')
subj_list.sort()
session_list = ['bl'] * len(subj_list)
time_step = 60
baseline_diagnosis_tau, total_av1451, remove_av1451 = removing_from_MRI_PET(subj_list, session_list, adni_path,
                                                                            mri_path,
                                                                            pet_path, get_subject_id, 'ADNI3',
                                                                            time_step, 36)
baseline_diagnosis_av45, total_av45, remove_av45 = removing_from_MRI_PET(subj_list, session_list, adni_path, mri_path,
                                                                         amy_path, get_subject_id, 'ADNI3', time_step,
                                                                         36)

baseline_diagnosis = pd.merge(baseline_diagnosis_tau, baseline_diagnosis_av45, how='inner')
total = pd.merge(total_av1451, total_av45, how='inner')
remove = pd.merge(remove_av1451, remove_av45, how='outer')

print(f'Baselien DX')
# baseline_diagnosis_av45.to_csv('/home/icml/Downloads/test.csv')
print(baseline_diagnosis["diagnosis"].value_counts())
print(f'total DX')
print(total["diagnosis"].value_counts())
print(f'remove DX')
print(remove["diagnosis"].value_counts())
baseline_diagnosis.to_csv('/home/icml/Downloads/adni2testmri/total_labels.csv')
total.to_csv('/home/icml/Downloads/adni2testmri/train_60d.csv')
remove.to_csv('/home/icml/Downloads/adni2testmri/remove_60d.csv'



################## extract demorgraphic ###########################
def extract_demorgraphic(csv_file):
    sex, age, mmse, cdr = csv_file['sex'], csv_file['age'], csv_file['MMSE'], csv_file['CDR']
    print(sex.value_counts())
    print(f'age : {np.round(np.mean(age),2)} | {np.round(np.std(age),2)}')
    print(f'MMSE : {np.round(np.mean(mmse),2)} | {np.round(np.std(mmse),2)}')
    print(f'CDR : {np.round(np.mean(cdr),2)} | {np.round(np.std(cdr),2)}')


total = pd.read_csv('/home/icml/Downloads/adni2testmri/train_60d.csv')
total_orig = total

total = total_orig[total_orig['diagnosis'] == 'CN']
extract_demorgraphic(total)

total = pd.concat([total_orig[total_orig['diagnosis'] == 'sMCI'], total_orig[total_orig['diagnosis'] == 'pMCI']])
extract_demorgraphic(total)

total = total_orig[total_orig['diagnosis'] == 'Dementia']
extract_demorgraphic(total)
################## extract demorgraphic ###########################
