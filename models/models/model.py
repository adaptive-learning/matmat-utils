import math
import pandas as pd


def sigmoid(x, c=0):
    return c + (1 - c) / (1 + math.exp(-x))


class Model:
    def __init__(self):
        self.logger = None

    def __str__(self):
        return "Not specified"

    def pre_process_data(self, data):
        pass

    def process_data(self, data):
        print "train",
        for answer in data.train_iter():
            prediction = self.predict(answer["student"], answer["item"], answer)
            self.update(answer["student"], answer["item"], prediction, answer["correct"], answer)

        print "test",
        for answer in data:
            prediction = self.predict(answer["student"], answer["item"], answer)
            self.update(answer["student"], answer["item"], prediction, answer["correct"], answer)
            if self.logger is not None:
                self.logger(answer, prediction)
            
    def predict(self, student, item, extra=None):
        pass

    def update(self, student, item, prediction, correct, extra=None):
        pass


class AvgModel(Model):
    def __init__(self):
        Model.__init__(self)
        self.corrects = 0
        self.all = 0

        self.avg = 0.5

    def __str__(self):
        return "Global average"

    def pre_process_data(self, data):
        pass

    def predict(self, student, item, extra=None):
        return self.avg

    def update(self, student, item, prediction, correct, extra=None):
        self.all += 1
        if correct:
            self.corrects += 1
        self.avg = float(self.corrects) / self.all


class AvgItemModel(Model):
    def __init__(self):
        Model.__init__(self)
        self.corrects = 0
        self.all = 0

    def __str__(self):
        return "Item average"

    def pre_process_data(self, data):
        items = data.get_items()
        self.corrects = pd.Series(index=items)
        self.counts = pd.Series(index=items)
        self.corrects[:] = 0
        self.counts[:] = 0

    def predict(self, student, item, extra=None):
        return self.corrects[item] / self.counts[item] if self.counts[item] > 0 else 0.5

    def update(self, student, item, prediction, correct, extra=None):
        self.counts[item] += 1
        if correct:
            self.corrects[item] += 1


