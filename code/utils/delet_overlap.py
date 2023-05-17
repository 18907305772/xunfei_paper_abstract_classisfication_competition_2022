import pandas as pd
import csv
import os


def get_dois(mode):
    raw_path = "../../xfdata/基于论文摘要的文本分类与查询性问答公开数据/"
    train_reader = csv.reader(open(raw_path + '{}.csv'.format(mode)), delimiter=',')
    next(train_reader)
    train_text_doi_map = dict()
    train_dois = []
    for line in train_reader:
        train_title = line[0].replace("\n", "").strip()
        train_abstract = line[3].replace("\n", "").strip()
        train_doi = line[4].replace("\n", "").strip()
        train_dois.append(train_doi)
        train_text_doi_map[train_title + train_abstract] = train_doi
    train_dois = set(train_dois)
    return train_dois, train_text_doi_map


def delete_9_1_train_valid(path_in, path_out, overlap_label_dois, text_doi_map):
    files = ["train.csv", "valid.csv"]
    for i in range(len(files)):
        data = pd.read_csv(path_in + files[i], delimiter=',')
        drop_row = []
        for row in range(data.shape[0]):
            text = data.loc[row, "sentence1"] + data.loc[row, "sentence2"] if not pd.isna(data.loc[row, "sentence2"]) else data.loc[row, "sentence1"]
            try:
                doi = text_doi_map[text]
                if doi in overlap_label_dois:
                    drop_row.append(row)
            except:
                print(i, row)
                raise KeyError
        data_drop = data.drop(drop_row)
        data_drop.to_csv(path_out + files[i], sep=',', index=False)
        print(len(drop_row), data.shape[0] - data_drop.shape[0])
    print("success")
    return


def delete_k_fold_train_valid(path_in, path_out, overlap_label_dois, text_doi_map):
    files = ["train.csv", "valid.csv"]
    for i in range(5):
        fold_path = "fold{}/".format(i)
        if not os.path.exists(path_out + fold_path):
            os.mkdir(path_out + fold_path)
        for j in range(len(files)):
            data = pd.read_csv(path_in + fold_path + files[j], delimiter=',')
            drop_row = []
            for row in range(data.shape[0]):
                text = data.loc[row, "sentence1"] + data.loc[row, "sentence2"] if not pd.isna(
                    data.loc[row, "sentence2"]) else data.loc[row, "sentence1"]
                try:
                    doi = text_doi_map[text]
                    if doi in overlap_label_dois:
                        drop_row.append(row)
                except:
                    print(i, row)
                    raise KeyError
            data_drop = data.drop(drop_row)
            data_drop.to_csv(path_out + fold_path + files[j], sep=',', index=False)
            print(len(drop_row), data.shape[0] - data_drop.shape[0])
        print("fold{} success".format(i))
    return


def main():
    train_dois, train_text_doi_map = get_dois("train")
    test_dois, _ = get_dois("test")
    overlap_label_dois = list(train_dois.intersection(test_dois))
    path_in_k_fold = "../../user_data/data_process-for-train_k-fold/"
    path_out_k_fold = path_in_k_fold[:-1] + "_drop-overlap/"
    if not os.path.exists(path_out_k_fold):
        os.mkdir(path_out_k_fold)
    delete_k_fold_train_valid(path_in_k_fold, path_out_k_fold, overlap_label_dois, train_text_doi_map)


if __name__ == "__main__":
    main()
