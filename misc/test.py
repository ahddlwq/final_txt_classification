# mem = Memory("./mycache")


# @mem.cache
# def get_data():
#     data = load_svmlight_file(FilePathConfig.train_feature_mat_path)
#     return data[0], data[1]
#
#
# X, y = get_data()
# print type(X), y
#
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
#
# data = SelectKBest(chi2, k=10000).fit_transform(X, y)
#
# from sklearn.datasets import dump_svmlight_file
#
# dump_svmlight_file(data, y, "labeled_chi2_fea.txt", False)

import math

# p = input()
# q = input()
# n = input()
p = 50
q = 20
n = 2
p = float(p) / 100
q = float(q) / 100

raw_result = 0
for i in range(1, n + 1):
    cur_p = float(p) / math.pow(2, i - 1)

    true_p = cur_p
    raw_result += true_p
    false_p = 1 - true_p
    x = 2
    while true_p <= 1:
        true_p += q
        raw_result += x * min(true_p, 1) * false_p
        false_p = (1 - true_p) * false_p
        x += 1
print raw_result
