import cPickle
import codecs

report = cPickle.load(open("../file/result_report-2017-04-05-13-31.pkl", 'r'))
confusion_matrix = report.confusion_matrix
labels = report.labels
file = codecs.open("asd.csv", "w", encoding="utf-8")

for i in range(confusion_matrix.shape[0]):
    file.write("," + labels[i][0:2])
file.write("\n")
for i in range(confusion_matrix.shape[0]):
    file.write(labels[i][0:2])
    for j in range(confusion_matrix.shape[1]):
        file.write("," + str(confusion_matrix[i][j]))
    file.write("\n")
file.close()
