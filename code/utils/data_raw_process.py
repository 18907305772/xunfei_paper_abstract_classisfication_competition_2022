"""
功能：将原始比赛训练、测试 csv 数据转换为训练、验证、和测试需要的 csv，提供给 train.py
输入：../../xfdata/基于论文摘要的文本分类与查询性问答公开数据/train.csv, test.csv
输出：
k-fold划分：../../user_data/data_process-for-train_k-fold/train.csv, valid.csv, test.csv
"""
import csv
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
import os
import re

RANDOM_SEED = 111


def get_only_chars(line):
    clean_line = ""
    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")  # replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if len(clean_line) == 0:
        return clean_line
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def data_raw_process(path_in, path_out, data_type='train', mode='one_fold', one_class_drop=False,
                     multi_space_to_one=False, multi_n_to_one=False, clean_label_or_aug_data=False, use_author=False,
                     use_doi=False, use_label=False):
    """
    :param path_in: 输入原始处理数据路径
    :param path_out: 输出路径
    :param data_type: 输入原始数据类型
    :param mode: one-fold 或者 k-fold 针对 data_type='train'
    :param one_class_drop: 丢弃只有一条数据的那个类 针对 data_type='train'
    :param multi_space_to_one: 把多个空格变成一个 针对 data_type='train' and 'test'
    :param multi_n_to_one: 把多个换行符换成一个换行符 针对 data_type='train' and 'test'
    :param clean_label_or_aug_data: 输入为清洗过 label 的或者增强后的数据 针对 data_type='train'
    :param use_author
    :param use_doi
    :param use_label
    :return: 最终要保存的 csv
    """
    assert data_type in ['train', 'test']
    assert mode in ['one_fold', 'k_fold']
    # 调整路径
    if one_class_drop is True:
        path_out = path_out[:-1] + "_one-class-drop/"
    if multi_space_to_one is True:
        path_out = path_out[:-1] + "_multi-space-to-one/"
    if multi_n_to_one is True:
        path_out = path_out[:-1] + "_multi-n-to-one/"
    if mode == "k_fold":
        path_out = path_out[:-1] + "_k-fold/"
    if clean_label_or_aug_data is True:
        path_out = path_out[:-1] + "_clean_label_or_aug/"
    if use_author is True:
        path_out = path_out[:-1] + "_use_author/"
    if use_doi is True:
        path_out = path_out[:-1] + "_use_doi/"
    if use_label is True:
        path_out = path_out[:-1] + "_use_label/"
    print("data path:{}".format(path_out))
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    path_final_in = path_in + data_type + '.csv'
    # 预处理输入文本
    reader = csv.reader(open(path_final_in), delimiter=',')
    next(reader)
    titles, abstracts = [], []
    authors = [] if use_author is True else None
    dois = [] if use_doi is True else None
    prefix_labels = [] if use_label is True else None
    nan_num = 0
    if data_type == 'train':
        labels = []
    for line in tqdm(reader):
        if clean_label_or_aug_data is True and data_type == 'train':
            title, abstract = line[0], line[1]
        else:
            title, abstract = line[0], line[3]
            author = line[1] if use_author is True else None
            doi = line[4] if use_doi is True else None
            prefix_label = 'Abdominal Fat, Artificial Intelligence, Culicidae, Diabetes Mellitus, Fasting, Gastrointestinal Microbiome, Inflammation, MicroRNAs, Neoplasms, Parkinson Disease, Psychology' if use_label is True else None
        if multi_n_to_one is True:
            title = re.sub('\n+', '\n', title).strip()
            abstract = re.sub('\n+', '\n', abstract).strip()
            author = re.sub('\n+', '\n', author).strip() if use_author is True else None
            doi = re.sub('\n+', '\n', doi).strip() if use_doi is True else None
            prefix_label = re.sub('\n+', '\n', prefix_label).strip() if use_label is True else None
        elif clean_label_or_aug_data is False:
            title = title.replace("\n", "").strip()
            abstract = abstract.replace("\n", "").strip()
            author = author.replace("\n", "").replace("[", "").replace("]", "").replace("\'", "").strip() if use_author is True else None
            doi = get_only_chars(doi) if use_doi is True else None
            prefix_label = prefix_label.replace("\n", "").strip() if use_label is True else None
        else:
            pass
        if multi_space_to_one is True:
            title = ' '.join(title.split())
            abstract = ' '.join(abstract.split())
            author = ' '.join(author.split()) if use_author is True else None
            doi = ' '.join(doi.split()) if use_doi is True else None
            prefix_label = ' '.join(prefix_label.split()) if use_label is True else None
        if data_type == 'train':
            if clean_label_or_aug_data is True and data_type == "train":
                label = line[2]
            else:
                label = line[5]
            if label == "":  # 为空的丢弃
                continue
            if title == "":
                print("title:", title)
                continue
            if abstract == "":
                print("abstract:", abstract)
                continue
            elif label == " Humboldt states":
                only_one_example = {"title": title, "abstract": abstract, "label": label}
                if use_author is True:
                    only_one_example["author"] = author
                if use_doi is True:
                    only_one_example["doi"] = doi
                if use_label:
                    only_one_example["prefix_label"] = prefix_label
                continue
            labels.append(label)
        titles.append(title)
        abstracts.append(abstract)
        if use_author is True:
            authors.append(author)
        if use_doi is True:
            dois.append(doi)
        if use_label is True:
            prefix_labels.append(prefix_label)
    # 写入训练/验证/测试集
    if data_type == 'train':
        # 划分训练验证集
        if mode == 'one_fold':
            args = [titles, abstracts, labels]
            if use_author:
                args.append(authors)
            if use_doi:
                args.append(dois)
            if use_label:
                args.append(prefix_labels)
            args = tuple(args)
            split_results = train_test_split(*args, train_size=0.9, random_state=RANDOM_SEED, shuffle=True, stratify=labels)
            train_titles, valid_titles, train_abstracts, valid_abstracts, train_labels, valid_labels = split_results[:6]
            if use_label:
                valid_prefix_labels = split_results.pop()
                train_prefix_labels = split_results.pop()
            if use_doi:
                valid_dois = split_results.pop()
                train_dois = split_results.pop()
            if use_author:
                valid_authors = split_results.pop()
                train_authors = split_results.pop()
            if one_class_drop is False and clean_label_or_aug_data is False:
                train_titles.append(only_one_example["title"])  # 把这个特殊的只有一个例子的加入训练集
                train_abstracts.append(only_one_example["abstract"])
                train_labels.append(only_one_example["label"])
                if use_author:
                    train_authors.append(only_one_example["author"])
                if use_doi:
                    train_dois.append(only_one_example["doi"])
                if use_label:
                    train_prefix_labels.append(only_one_example["prefix_label"])
            train_data = [train_titles, train_abstracts, train_labels]
            valid_data = [valid_titles, valid_abstracts, valid_labels]
            if use_author:
                train_data.append(train_authors)
                valid_data.append(valid_authors)
            if use_doi:
                train_data.append(train_dois)
                valid_data.append(valid_dois)
            if use_label:
                train_data.append(train_prefix_labels)
                valid_data.append(valid_prefix_labels)
            data = [train_data, valid_data]
            for idx, file_name in enumerate(['train.csv', 'valid.csv']):
                path_final_out = path_out + file_name
                writer = csv.writer(open(path_final_out, 'w'), delimiter=',')
                head = ['sentence1', 'sentence2']
                i = 3
                if use_author:
                    head.append('sentence{}'.format(i))
                    i += 1
                if use_doi:
                    head.append('sentence{}'.format(i))
                    i += 1
                if use_label:
                    head.append('sentence{}'.format(i))
                    i += 1
                head.append('label')
                writer.writerow(head)
                all_data = (data[idx][j] for j in range(len(head)))
                for line in zip(*all_data):
                    line = list(line)  # [title, abstract, label, (author, dos, prefix_label)]
                    gt_label = line[2]
                    del line[2]
                    line.append(gt_label)
                    writer.writerow(line)
        elif mode == 'k_fold':
            if use_author or use_doi or use_label is True:
                raise NotImplementedError
            kf = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED, shuffle=True)
            titles_and_abstracts = [(titles[i], abstracts[i]) for i in range(len(titles))]
            for i, (train_index, test_index) in enumerate(kf.split(titles_and_abstracts, labels)):
                print(f'StratifiedKFold {i + 1}:')
                train_titles_and_abstracts, valid_titles_and_abstracts = np.array(titles_and_abstracts)[train_index], \
                                                                         np.array(titles_and_abstracts)[test_index]
                train_titles_and_abstracts = train_titles_and_abstracts.tolist()
                valid_titles_and_abstracts = valid_titles_and_abstracts.tolist()
                train_titles = [train_titles_and_abstracts[i][0] for i in range(len(train_titles_and_abstracts))]
                train_abstracts = [train_titles_and_abstracts[i][1] for i in range(len(train_titles_and_abstracts))]
                valid_titles = [valid_titles_and_abstracts[i][0] for i in range(len(valid_titles_and_abstracts))]
                valid_abstracts = [valid_titles_and_abstracts[i][1] for i in range(len(valid_titles_and_abstracts))]
                train_labels, valid_labels = np.array(labels)[train_index], np.array(labels)[test_index]
                train_labels = train_labels.tolist()
                valid_labels = valid_labels.tolist()
                if one_class_drop is False and clean_label_or_aug_data is False:
                    train_titles.append(only_one_example["title"])  # 把这个特殊的只有一个例子的加入训练集
                    train_abstracts.append(only_one_example["abstract"])
                    train_labels.append(only_one_example["label"])
                train_data = [train_titles, train_abstracts, train_labels]
                valid_data = [valid_titles, valid_abstracts, valid_labels]
                data = [train_data, valid_data]
                path_final_out = path_out + 'fold{}/'.format(i)
                if not os.path.exists(path_final_out):
                    os.mkdir(path_final_out)
                for idx, file_name in enumerate(['train.csv', 'valid.csv']):
                    file_final_out = path_final_out + file_name
                    writer = csv.writer(open(file_final_out, 'w'), delimiter=',')
                    writer.writerow(['sentence1', 'sentence2', 'label'])
                    for line in zip(data[idx][0], data[idx][1], data[idx][2]):
                        line = list(line)
                        writer.writerow(line)

    elif data_type == 'test':
        test_data = [titles, abstracts]
        if use_author:
            test_data.append(authors)
        if use_doi:
            test_data.append(dois)
        if use_label:
            test_data.append(prefix_labels)
        path_final_out = path_out + 'test.csv'
        writer = csv.writer(open(path_final_out, 'w'), delimiter=',')
        head = ['sentence1', 'sentence2']
        i = 3
        if use_author:
            head.append('sentence{}'.format(i))
            i += 1
        if use_doi:
            head.append('sentence{}'.format(i))
            i += 1
        if use_label:
            head.append('sentence{}'.format(i))
            i += 1
        head.append('label')
        writer.writerow(head)
        all_test_data = (test_data[j] for j in range(len(head) - 1))
        for line in zip(*all_test_data):
            line = list(line)  # [title, abstract, (author, dos, prefix_label)]
            line.append(-1)
            if clean_label_or_aug_data is True:  # 对于增强数据，同样请理测试集
                line[0] = get_only_chars(line[0])
                line[1] = get_only_chars(line[1])
            writer.writerow(line)
    return


def main():
    path_in = '../../xfdata/基于论文摘要的文本分类与查询性问答公开数据/'
    path_out = '../../user_data/data_process-for-train/'
    data_raw_process(path_in, path_out, data_type='train', mode='k_fold', one_class_drop=False,
                     multi_space_to_one=False, multi_n_to_one=False, clean_label_or_aug_data=False, use_author=False,
                     use_doi=False, use_label=False)
    data_raw_process(path_in, path_out, data_type='test', mode='k_fold', one_class_drop=False,
                     multi_space_to_one=False, multi_n_to_one=False, clean_label_or_aug_data=False, use_author=False,
                     use_doi=False, use_label=False)


if __name__ == "__main__":
    main()
