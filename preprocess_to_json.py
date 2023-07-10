import os
import json
import glob
import copy
import pandas as pd
from pathlib import Path
from config import ex, config
from text_norm import normalize_text
from collections import defaultdict
from sklearn.model_selection import train_test_split

def stat_diseases(df):
    # Initialize a defaultdict of type int
    diseases_all = defaultdict(list)
    diseases_num = defaultdict(int)

    # Iterate over each row in the 'Problems' column
    for index, row in df.iterrows():
        # Check if 'i' is a string
        i = row["Problems"]
        if isinstance(i, str):
            if ";" in i:
                # Split the string and iterate over each disease
                disease = i.split(";")
                for j in disease:
                    if "," in j:
                        disease_minor = j.split(",")
                        for k in disease_minor:
                            # diseases_all[k.strip()].append(row["uid"])
                            diseases_all[k.strip()].append(row["filename"])
                            diseases_num[k.strip()]+=1
                    else: 
                        # diseases_all[j.strip()].append(row["uid"])
                        diseases_all[j.strip()].append(row["filename"])
                        diseases_num[j.strip()]+=1
            elif "," in i:
                disease = i.split(",")
                for j in disease:
                    # diseases_all[j.strip()].append(row["uid"])
                    diseases_all[j.strip()].append(row["filename"])
                    diseases_num[j.strip()]+=1

            else:
                # diseases_all[i.strip()].append(row["uid"])
                diseases_all[i.strip()].append(row["filename"])
                diseases_num[i.strip()]+=1
    return diseases_all, diseases_num

def balance_dataset(df):
    diseases_all, diseases_num = stat_diseases(df)
    
    rows_out = []
    for filename in diseases_all["normal"][:500]:
        row = df[df['filename'] == filename].iloc[0].to_dict()
        rows_out.append(row)
    
    for problem in diseases_all.keys():
        if problem != "normal":
            for filename in diseases_all[problem]:
                row = df[df['filename'] == filename].iloc[0].to_dict()
                rows_out.append(row)
    
    df_out = pd.DataFrame(rows_out)
    return df_out
        
def save_dp_as_json(df_dataset, out_dir, split='train'):
    """
    split: {'train', 'valid'}
    """
    if split == 'train':
        df_dataset = balance_dataset(df_dataset)
    normal_count = df_dataset['Problems'].apply(lambda x: "normal" in str(x).lower()).sum()
    abnormal_count = df_dataset['Problems'].apply(lambda x: "normal" not in str(x).lower()).sum()
    print("length of raw data", len(df_dataset))
    print("length of processed data", len(df_dataset))
    print("normal_count, abnormal_count", normal_count, abnormal_count)
    
    findings_fp = os.path.join(out_dir, "findings_{}.json".format(split))
    with open(findings_fp, 'w') as f:
        df_dataset_findings = df_dataset.copy()
        df_dataset_findings['caption'] = df_dataset_findings['findings']
        df_dataset_findings = df_dataset_findings[df_dataset_findings['caption'].notna()].reset_index(drop=True)
        df_dataset_findings['caption'] = df_dataset_findings['caption'].apply(normalize_text)
        json.dump(df_dataset_findings[['img_fp', 'image_id', 'caption']].to_dict('records'), f, ensure_ascii=False)
    print("file output to " + findings_fp)

    impression_fp = os.path.join(out_dir, "impression_{}.json".format(split))
    with open(impression_fp, 'w') as f:
        df_dataset_impression = df_dataset.copy()
        df_dataset_impression['caption'] = df_dataset_impression['impression']
        df_dataset_impression = df_dataset_impression[df_dataset_impression['caption'].notna()].reset_index(drop=True)
        df_dataset_impression['caption'] = df_dataset_impression['caption'].apply(normalize_text)
        json.dump(df_dataset_impression[['img_fp', 'image_id', 'caption']].to_dict('records'), f, ensure_ascii=False)
    print("file output to " + impression_fp)

@ex.automain  
def main(_config):
    _config = config()
    _config = copy.deepcopy(_config)
    indiana_dir = _config['indiana_dir']
    out_dir = _config['preprocess_dir']
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    img_fps_all = glob.glob(os.path.join(indiana_dir, "images", "images_normalized", "*.png"))
    img_fps_all_mapping = {os.path.basename(f): f for f in img_fps_all}

    # keeo image with both frontal and lateral projections
    df_proj = pd.read_csv(os.path.join(indiana_dir, "indiana_projections.csv"))
    # df_proj = df_proj[df_proj["projection"].apply(lambda x: ("lateral" in x.lower()))] # df_proj = df_proj[df_proj['projection'] == 'Frontal'].reset_index(drop=True)
    valid_uids = df_proj.groupby('uid')['projection'].apply(list).reset_index()
    valid_uids = valid_uids.loc[valid_uids['projection'].apply(len) == 2, 'uid']
    df_proj = df_proj[df_proj['uid'].isin(valid_uids)].reset_index(drop=True)
    df_proj['img_fp'] = df_proj['filename'].apply(img_fps_all_mapping.get)
    df_proj['image_id'] = df_proj.index

    # keep image with both findings and impression
    df_cap = pd.read_csv(os.path.join(indiana_dir, "indiana_reports.csv")) [['uid', 'findings', 'impression', "Problems"]]
    df_cap = df_cap[df_cap['findings'].notna() & df_cap['impression'].notna()].reset_index(drop=True)
    assert (df_cap.groupby('uid')['uid'].apply(len) > 1).sum() == 0, "with double caption for the same image"

    df_dataset_raw = pd.merge(df_proj, df_cap, on='uid', how='left')
    # df_dataset_raw.to_csv(os.path.join(out_dir, "df_dataset_raw.csv"))

    df_dataset_train, df_dataset_valid = train_test_split(df_dataset_raw, test_size=0.1, random_state=_config['seed'])
    save_dp_as_json(df_dataset_train, out_dir, split='train')
    save_dp_as_json(df_dataset_valid, out_dir, split='valid')
