import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np

path = '/home/icml/Downloads/adni2testmri/train_60d.csv'
save_path = '/home/icml/Downloads/adni2testmri/getlabels_save_1data'
data_df = pd.read_csv(path)

skf = StratifiedKFold(shuffle=True, random_state=77)
X = data_df['participant_id']
y = data_df['diagnosis']

skf.get_n_splits(X, y)
print(skf)
fold = 0
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if not os.path.exists(os.path.join(save_path, "train_splits-5", f"split-{fold}")):
        os.makedirs(os.path.join(save_path, "train_splits-5", f"split-{fold}"))
    if not os.path.exists(os.path.join(save_path, "validation_splits-5", f"split-{fold}")):
        os.makedirs(os.path.join(save_path, "validation_splits-5", f"split-{fold}"))
    result = pd.DataFrame(
        {'participant_id': X_train, 'session_id': 'nan', 'diagnosis': y_train, 'age': 'nan', 'sex': 'nan'})

    result = result.sample(frac=1).reset_index(drop=True)

    ad_subj = result[result['diagnosis'] == 'Dementia'].reset_index(drop=True)
    smci_subj = result[result['diagnosis'] == 'sMCI'].reset_index(drop=True)
    pmci_subj = result[result['diagnosis'] == 'pMCI'].reset_index(drop=True)
    mci_subj = pd.concat([smci_subj, pmci_subj]).reset_index(drop=True)
    mci_subj['diagnosis'] = mci_subj['diagnosis'].map({'sMCI': 'MCI', 'pMCI': 'MCI'})
    cn_subj = result[result['diagnosis'] == 'CN'].reset_index(drop=True)
    ad_subj.to_csv(os.path.join(save_path, "train_splits-5", f"split-{fold}", "./AD.tsv"), sep='\t', index=False)
    smci_subj.to_csv(os.path.join(save_path, "train_splits-5", f"split-{fold}", "sMCI.tsv"), sep='\t', index=False)
    pmci_subj.to_csv(os.path.join(save_path, "train_splits-5", f"split-{fold}", "pMCI.tsv"), sep='\t', index=False)
    mci_subj.to_csv(os.path.join(save_path, "train_splits-5", f"split-{fold}", "./MCI.tsv"), sep='\t', index=False)
    cn_subj.to_csv(os.path.join(save_path, "train_splits-5", f"split-{fold}", "./CN.tsv"), sep='\t', index=False)

    result = pd.DataFrame(
        {'participant_id': X_test, 'session_id': 'nan', 'diagnosis': y_test, 'age': 'nan', 'sex': 'nan'})
    ad_subj = result[result['diagnosis'] == 'Dementia'].reset_index(drop=True)
    smci_subj = result[result['diagnosis'] == 'sMCI'].reset_index(drop=True)
    pmci_subj = result[result['diagnosis'] == 'pMCI'].reset_index(drop=True)
    mci_subj = pd.concat([smci_subj, pmci_subj])
    mci_subj['diagnosis'] = mci_subj['diagnosis'].map({'sMCI': 'MCI', 'pMCI': 'MCI'})
    cn_subj = result[result['diagnosis'] == 'CN'].reset_index(drop=True)
    ad_subj.to_csv(os.path.join(save_path, "validation_splits-5", f"split-{fold}", "AD_baseline.tsv"), sep='\t',
                   index=False)
    smci_subj.to_csv(os.path.join(save_path, "validation_splits-5", f"split-{fold}", "sMCI_baseline.tsv"), sep='\t',
                     index=False)
    mci_subj.to_csv(os.path.join(save_path, "validation_splits-5", f"split-{fold}", "MCI_baseline.tsv"), sep='\t',
                    index=False)
    pmci_subj.to_csv(os.path.join(save_path, "validation_splits-5", f"split-{fold}", "pMCI_baseline.tsv"), sep='\t',
                     index=False)
    cn_subj.to_csv(os.path.join(save_path, "validation_splits-5", f"split-{fold}", "CN_baseline.tsv"), sep='\t',
                   index=False)
    fold += 1


# ad_subj = data_df[data_df['diagnosis'] == 'Dementia']


def check_mmse_cdr(cn_subj, cn_session):
    adnimerge = '/home/icml/Downloads/ADNI_DATA/ADNIMERGE.csv'
    adnimerge = pd.read_csv(adnimerge)
    cn_df = pd.DataFrame(columns=['MMSE', 'CDR'])
    for c in range(len(cn_subj)):
        cn_subj[c] = cn_subj[c][8:11] + '_S_' + cn_subj[c][12:16]
        tmp = adnimerge[adnimerge['PTID'] == cn_subj[c]]
        if cn_session[c] == 'ses-M00':
            cn_session[c] = 'bl'
        elif len(cn_session[c][5:]) == 3:
            cn_session[c] = 'm' + cn_session[c][5] + cn_session[c][6] + cn_session[c][7]
        else:
            cn_session[c] = 'm' + cn_session[c][5] + cn_session[c][6]
        # print(cn_session[c])
        tmp = tmp[tmp['VISCODE'] == cn_session[c]]
        tmp_mmse = tmp['MMSE']
        tmp_cdr = tmp['CDRSB']
        tmp = pd.DataFrame({'MMSE': tmp_mmse, 'CDR': tmp_cdr})
        cn_df = pd.concat([cn_df, tmp])
    return cn_df


output_path = '/home/icml/Downloads/adni2testmri/train_60d.csv'
df = pd.read_csv(output_path)

cn = df[df['diagnosis'] == 'CN']
mci = pd.merge(df[df['diagnosis'] == 'sMCI'], df[df['diagnosis'] == 'sMCI'])
ad = df[df['diagnosis'] == 'Dementia']

# cn_df = check_mmse_cdr(list(cn['participant_id']), list(cn['session']))
# mci_df = check_mmse_cdr(list(mci['participant_id']), list(mci['session']))
# ad_df = check_mmse_cdr(list(ad['participant_id']), list(ad['session']))
print('CN')
print(cn['sex'].value_counts())
cn_age = list(cn['age'].dropna())
cn_mmse = cn['MMSE']
cn_cdr = cn['CDR']
print(np.round(np.mean(cn_age), 2), np.round(np.std(cn_age), 2))
print(np.round(np.mean(cn_mmse), 2), np.round(np.std(cn_mmse), 2))
print(np.round(np.mean(cn_cdr), 2), np.round(np.std(cn_cdr), 2))

print('MCI')
print(mci['sex'].value_counts())
mci_age = list(mci['age'].dropna())
mci_mmse = mci['MMSE']
mci_cdr = mci['CDR']
print(np.round(np.mean(mci_age), 2), np.round(np.std(mci_age), 2))
print(np.round(np.mean(mci_mmse), 2), np.round(np.std(mci_mmse), 2))
print(np.round(np.mean(mci_cdr), 2), np.round(np.std(mci_cdr), 2))

print('AD')
print(ad['sex'].value_counts())
ad_age = list(ad['age'])
ad_mmse = ad['MMSE']
ad_cdr = ad['CDR']
print(np.round(np.mean(ad_age), 2), np.round(np.std(ad_age), 2))
print(np.round(np.mean(ad_mmse), 2), np.round(np.std(ad_mmse), 2))
print(np.round(np.mean(ad_cdr), 2), np.round(np.std(ad_cdr), 2))
