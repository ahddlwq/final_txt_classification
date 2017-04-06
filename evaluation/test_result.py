from datetime import datetime

from sklearn import metrics

from config.config import FilePathConfig, ClassifierConfig
from misc.util import Util

class TestResult(object):
    def __init__(self, predicted_class, raw_class_label, labels):
        self.predicted_class = predicted_class
        self.raw_class_label = raw_class_label
        self.labels = labels
        pass

    # def __str__(self):
    #     return "Test set size: " + str(self.test_size) + "\n" \
    #            + "[MacroAverage]: " + " Precision: " + str(self.macro_precision) + " Recall: " + str(
    #         self.macro_recall) + " FMeasure: " + str(self.macro_fmeasure) + "\n" \
    #            + "[MicroAverage]: " + str(self.micro_average)

    def print_report(self):
        predicted_class = self.predicted_class
        raw_class_label = self.raw_class_label
        self.macro_precision = metrics.precision_score(raw_class_label, predicted_class, average="macro")
        self.macro_recall = metrics.recall_score(raw_class_label, predicted_class, average="macro")
        self.classification_report = metrics.classification_report(raw_class_label, predicted_class,
                                                                   target_names=self.labels)
        self.confusion_matrix = metrics.confusion_matrix(raw_class_label, predicted_class)

        Util.log_tool.log.info(self.classification_report.encode(FilePathConfig.file_encodeing))
        Util.log_tool.log.info(self.confusion_matrix)
        self.save_report()

    def save_report(self):
        time = datetime.now().strftime("-%Y-%m-%d-%H-%M")
        label = time + '-' + ClassifierConfig.cur_single_model
        Util.save_object_into_pkl(self, str(FilePathConfig.result_report_path) % label)
