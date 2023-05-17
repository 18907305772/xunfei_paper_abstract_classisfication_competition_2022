cd utils
python data_raw_process.py
python delet_overlap.py
cd ..
cp ../user_data/data_process-for-train_k-fold/test.csv ../user_data/data_process-for-train_k-fold_drop-overlap/test.csv