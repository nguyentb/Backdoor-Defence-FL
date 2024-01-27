import pandas as pd
from glob import glob
from chestXRay import data_dir1


data_dir2 = '.\\input\\chestxray8-dataframe\\'
# df = pd.read_csv(data_dir1 + 'Data_Entry_2017.csv')
df = pd.read_csv(data_dir1 + 'Data_Entry_2017.csv')
image_label_map = pd.read_csv(data_dir2 + 'train_df.csv')
bad_labels = pd.read_csv(data_dir2 + 'cxr14_bad_labels.csv')

# Listing all the .jpg filepaths
image_paths = glob(data_dir1 + 'images_*\\images\\*.png')

# print(f'Total image files found : {len(image_paths)}')
# print(f'Total number of image labels: {image_label_map.shape[0]}')
# print(f'Unique patients: {len(df["Patient ID"].unique())}')

labels = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',
          'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
          'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema',
          'Consolidation', 'Healthy']

df.rename(columns={"Image Index": "Index"}, inplace=True)
image_label_map.rename(columns={"Image Index": "Index"}, inplace=True)

df.drop(df.iloc[:, 2:], inplace=True, axis=1)

print("Splitting the entries with more than one label into individual entries.")
# new_frames = [df]
for i in range(len(df['Finding Labels'])):
    index_labels = df['Finding Labels'][i].split('|')
    if len(index_labels) > 1:
        index_val = df.Index[i]
        df = df[~(df.Index == index_val)]

        # for l in index_labels:
        #     new_frames.append(pd.DataFrame({'Index': index_val, 'Finding Labels': l}, index=[0]))
            # df = pd.concat([new_row, df.loc[:]]).reset_index(drop=True)

# df = df[~df.Index.isin(double_labels.keys())]
# df = pd.concat(new_frames).reset_index(drop=True)
df = df.reset_index(drop=True)
# removing bad labels
print("Removing bad labels")
df = df[~df.Index.isin(bad_labels.Index)]

for c_label in labels:
    df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

# print(df)

# remove labels with too few cases
print("Removing labels with too few cases")
MIN_CASES = 1000
labels = [c_label for c_label in labels if (df[c_label].sum() > MIN_CASES)]

df = df[(df["Finding Labels"].isin(labels))]

print('Clean Labels ({})'.format(len(labels)),
      [(c_label, int(df[c_label].sum())) for c_label in labels])

# print(df)

MAX_CASES = 3000
print("Keeping only a maximum of",str(MAX_CASES), "entries for each label")
df = df.groupby("Finding Labels").head(MAX_CASES)

print('Clean Labels ({})'.format(len(labels)),
      [(c_label, int(df[c_label].sum())) for c_label in labels])
# print(df)

df['disease_vec'] = df.apply(lambda x: [x[labels].values], 1).map(lambda x: x[0])

df.drop(df.iloc[:, 2:-1], inplace=True, axis=1)

Index = []
for path in image_paths:
    Index.append(path.split('\\')[-1])
index_path_map = pd.DataFrame({'Index': Index, 'path': image_paths})

# Merge the absolute path of the images to the main dataframe
df = pd.merge(df, index_path_map, on='Index', how='inner')

df.to_csv("clean_dataframe_small", encoding='utf-8', index=False)
