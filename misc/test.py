from sklearn.datasets import load_svmlight_file

from config.config import FilePathConfig


# mem = Memory("./mycache")


# @mem.cache
def get_data():
    data = load_svmlight_file(FilePathConfig.train_feature_mat_path)
    return data[0], data[1]


X, y = get_data()
print type(X), y

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

data = SelectKBest(chi2, k=10000).fit_transform(X, y)

from sklearn.datasets import dump_svmlight_file

dump_svmlight_file(data, y, "labeled_chi2_fea.txt", False)
