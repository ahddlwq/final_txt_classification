class TestResult(object):
    def __init__(self):
        self.micro_average = 0
        self.macro_precision = 0
        self.macro_recall = 0
        self.macro_fmeasure = 0
        self.test_size = 0
        pass

    def __str__(self):
        return "Test set size: " + str(self.test_size) + "\n" \
               + "[MacroAverage]: " + " Precision: " + str(self.macro_precision) + " Recall: " + str(
            self.macro_recall) + " FMeasure: " + str(self.macro_fmeasure) + "\n" \
               + "[MicroAverage]: " + str(self.micro_average)

    def evaluation(self, predicted_class, raw_class_label):
        pass
