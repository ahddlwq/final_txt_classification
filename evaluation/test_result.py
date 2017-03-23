class TestResult(object):
    def __init__(self):
        self.micro_average = 0
        self.macro_precision = 0
        self.macro_recall = 0
        self.macro_fmeasure = 0
        self.test_size = 0
        pass

    def __str__(self):
        return "Test set size: " + self.test_size + "\n" \
               + "[MacroAverage]: " + " Precision: " + self.macro_precision + " Recall: " + self.macro_recall + " FMeasure: " + self.macro_fmeasure + "\n" \
               + "[MicroAverage]: " + self.micro_average;
