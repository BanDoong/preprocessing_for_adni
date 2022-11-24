import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from sklearn import model_selection as sk_model_selection
path = '/home/icml/Desktop/coding_test/kongsil/train_labels.csv'
save_path = '/home/icml/Desktop/coding_test/kongsil'
df = pd.read_csv(path)

X = df['BraTS21ID']
y = df['MGMT_value']
skf = StratifiedKFold(shuffle=True, random_state=77)
skf.get_n_splits(X, y)

train_df = pd.read_csv("/home/icml/Desktop/coding_test/kongsil/train_labels.csv")
# display(train_df)

df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)

df_train.to_csv(os.path.join(save_path, 'train.csv'))
df_valid.to_csv(os.path.join(save_path, 'test.csv'))

# fold = 0
# for train_idx, test_idx in skf.split(X, y):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#     if not os.path.exists(os.path.join(save_path, "train_splits-5", f"split-{fold}")):
#         os.makedirs(os.path.join(save_path, "train_splits-5", f"split-{fold}"))
#     if not os.path.exists(os.path.join(save_path, "validation_splits-5", f"split-{fold}")):
#         os.makedirs(os.path.join(save_path, "validation_splits-5", f"split-{fold}"))
#     train = pd.DataFrame({'participant_id': X_train, 'diagnosis': y_train})
#     test = pd.DataFrame({'participant_id': X_train, 'diagnosis': y_train})
#
#     train.to_csv(os.path.join(save_path, "train_splits-5", f"split-{fold}", f"train_{fold}.csv"))
#     test.to_csv(os.path.join(save_path, "validation_splits-5", f"split-{fold}", f"test_{fold}.csv"))
#     fold += 1

