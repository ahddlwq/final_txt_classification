from config.config import FilePathConfig
from util.util import Util

# lr_path = FilePathConfig.file_root_path + "lr-raw_results.txt"
# svm_path = FilePathConfig.file_root_path + "svm-raw_results.txt"
# xgb_path = FilePathConfig.file_root_path + "xgb-raw_results.txt"
#
# lr_results = Util.load_object_from_pkl(lr_path)
# svm_results = Util.load_object_from_pkl(svm_path)
# xgb_results = Util.load_object_from_pkl(xgb_path)
#
# length = len(lr_results)
# result = []
#
#
# for i in xrange(length):
#     print i
#     lr_result = lr_results[i][0][0]
#     svm_result = svm_results[i][0][0]
#     xgb_result = xgb_results[i][0][0]
#     if lr_result == xgb_result and svm_result == xgb_result:
#         result.append((i, lr_result, 3, lr_result, svm_result, xgb_result))
#     elif lr_result == xgb_result or svm_result == xgb_result:
#         result.append((i, xgb_result, 2, lr_result, svm_result, xgb_result))
#     elif lr_result == svm_result:
#         result.append((i, lr_result, 1, lr_result, svm_result, xgb_result))
#     else:
#         result.append((i, svm_result, 0, lr_result, svm_result, xgb_result))
#
# Util.save_object_into_pkl(result, FilePathConfig.file_root_path + "result2.pkl")

result = Util.load_object_from_pkl(FilePathConfig.file_root_path + "result2.pkl")
weight_dic = {}
weight_dic[3] = 0
weight_dic[2] = 0
weight_dic[1] = 0
weight_dic[0] = 0

class_dic = {}
for line in result:
    weight = line[2]
    weight_dic[weight] += 1
    cla = line[1]
    if weight >= 1:
        if class_dic.get(cla) == None:
            class_dic[cla] = 1
        class_dic[cla] += 1

for key, value in weight_dic.items():
    print key, value

for key, value in class_dic.items():
    print key, value
