"""
功能：投票融合 k-fold 模型、多模型输出文件
输入：
输出：
"""
import csv
import collections
from tqdm import tqdm
import numpy as np


def softmax(x):
    """ softmax function """
    # assert(len(x.shape) > 1, "dimension must be larger than 1")
    # print(np.max(x, axis = 1, keepdims = True)) # axis = 1, 行
    x -= np.max(x, axis=1, keepdims=True)  # 为了稳定地计算softmax概率， 一般会减掉最大的那个元素
    x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return x


class Model_ensemble(object):
    def __init__(self):
        self.test_size = 2086

    def k_fold_voting(self, k_fold_input_files, output_file, k=5):
        assert k == len(k_fold_input_files)
        result = []
        readers = []
        count = []
        for i in range(k):
            reader = csv.reader(open(k_fold_input_files[i]), delimiter=',')
            next(reader)
            reader = [line[0] for line in reader]
            readers.append(reader)

        for i in tqdm(range(self.test_size)):  # 遍历每个 test exp
            exp_res = []
            for j in range(k):  # 遍历其所有 fold 预测结果
                exp_line = readers[j][i]
                exp_res.append(exp_line)
            pred_counter = collections.Counter(exp_res)
            result.append(pred_counter.most_common(1)[0][0])  # 取最多的作为输出
            count.append(pred_counter.most_common(1)[0][1])

        writer = csv.writer(open(output_file, 'w'), delimiter=',')
        writer.writerow(['Topic(Label)'])
        for item in result:
            writer.writerow([item])
        print("Finish {}-fold ensemble, output file is in: {}".format(k, output_file))
        return

    def multi_model_voting(self, multi_model_input_files, output_file, models_weight=None):
        print("num input files:{}".format(len(multi_model_input_files)))
        result = []
        readers = []
        for i in range(len(multi_model_input_files)):
            reader = csv.reader(open(multi_model_input_files[i]), delimiter=',')
            next(reader)
            reader = [line[0] for line in reader]
            readers.append(reader)

        for i in tqdm(range(self.test_size)):  # 遍历每个 test exp
            if models_weight is not None:
                exp_res = dict()
            else:
                exp_res = []
            for j in range(len(multi_model_input_files)):  # 遍历其所有模型预测结果
                exp_line = readers[j][i]
                if models_weight is not None:
                    exp_res.setdefault(exp_line, 0)
                    exp_res[exp_line] += 1 * models_weight[j]
                else:
                    exp_res.append(exp_line)
            pred_counter = collections.Counter(exp_res)
            result.append(pred_counter.most_common(1)[0][0])  # 取最多的作为结果

        writer = csv.writer(open(output_file, 'w'), delimiter=',')
        writer.writerow(['Topic(Label)'])
        for item in result:
            writer.writerow([item])
        print("Finish multi model ensemble, output file is in: {}".format(output_file))
        return


def main():
    print("--------------- k-fold ensemble ---------------")
    k_fold_input_files_1 = ["submit/drop-overlap_fold{}_michiyasunaga_BioLinkBERT-large_predict-top2.csv".format(i) for
                            i in
                            range(5)]
    output_file_1 = "submit/ensemble_k-fold_data_process-for-train_k-fold_drop-overlap_michiyasunaga_BioLinkBERT-large_predict-top2.csv"
    model_ensemble_1 = Model_ensemble()
    model_ensemble_1.k_fold_voting(k_fold_input_files_1, output_file_1, k=5)

    k_fold_input_files_2 = ["submit/drop-overlap_fold{}_dmis-lab_biobert-base-cased-v1.2_predict-top2.csv".format(i) for
                            i in
                            range(5)]
    output_file_2 = "submit/ensemble_k-fold_data_process-for-train_k-fold_drop-overlap_dmis-lab_biobert-base-cased-v1.2_predict-top2.csv"
    model_ensemble_2 = Model_ensemble()
    model_ensemble_2.k_fold_voting(k_fold_input_files_2, output_file_2, k=5)

    k_fold_input_files_3 = ["submit/fold{}_michiyasunaga_BioLinkBERT-large_predict-top2.csv".format(i) for i in
                            range(5)]
    output_file_3 = "submit/ensemble_k-fold_data_process-for-train_k-fold_michiyasunaga_BioLinkBERT-large_predict-top2.csv"
    model_ensemble_3 = Model_ensemble()
    model_ensemble_3.k_fold_voting(k_fold_input_files_3, output_file_3, k=5)

    k_fold_input_files_4 = ["submit/fold{}_dmis-lab_biobert-base-cased-v1.2_predict-top2.csv".format(i) for i in
                            range(5)]
    output_file_4 = "submit/ensemble_k-fold_data_process-for-train_k-fold_dmis-lab_biobert-base-cased-v1.2_predict-top2.csv"
    model_ensemble_4 = Model_ensemble()
    model_ensemble_4.k_fold_voting(k_fold_input_files_4, output_file_4, k=5)

    k_fold_input_files_5 = [
        "submit/fold{}_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_predict-top2.csv".format(i) for i in
        range(5)]
    output_file_5 = "submit/ensemble_k-fold_data_process-for-train_k-fold_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_predict-top2.csv"
    model_ensemble_5 = Model_ensemble()
    model_ensemble_5.k_fold_voting(k_fold_input_files_5, output_file_5, k=5)

    k_fold_input_files_6 = ["submit/fold{}_microsoft_deberta-base_predict-top2.csv".format(i) for i in
                            range(5)]
    output_file_6 = "submit/ensemble_k-fold_data_process-for-train_k-fold_microsoft_deberta-base_predict-top2.csv"
    model_ensemble_6 = Model_ensemble()
    model_ensemble_6.k_fold_voting(k_fold_input_files_6, output_file_6, k=5)

    print("--------------- multi model ensemble ---------------")
    multi_model_input_files = [
        "submit/ensemble_k-fold_data_process-for-train_k-fold_drop-overlap_michiyasunaga_BioLinkBERT-large_predict-top2.csv",
        "submit/ensemble_k-fold_data_process-for-train_k-fold_drop-overlap_dmis-lab_biobert-base-cased-v1.2_predict-top2.csv",
        "submit/ensemble_k-fold_data_process-for-train_k-fold_michiyasunaga_BioLinkBERT-large_predict-top2.csv",
        "submit/ensemble_k-fold_data_process-for-train_k-fold_dmis-lab_biobert-base-cased-v1.2_predict-top2.csv",
        "submit/ensemble_k-fold_data_process-for-train_k-fold_microsoft_BiomedNLP-PubMedBERT-base-uncased-abstract_predict-top2.csv",
        "submit/ensemble_k-fold_data_process-for-train_k-fold_microsoft_deberta-base_predict-top2.csv",
    ]

    output_file_final = "../prediction_result/result.csv"
    model_ensemble_final = Model_ensemble()
    model_weights = [[0.94151, 0.93912, 0.93864, 0.93864, 0.93672, 0.93672]]
    model_weights = softmax(np.array(model_weights))[0].tolist()
    model_ensemble_final.multi_model_voting(multi_model_input_files, output_file_final, models_weight=model_weights)


if __name__ == "__main__":
    main()
