import datetime
from data.data import *
import pandas as pd
from hashlib import sha1


def get_hash(model, data):
    return sha1(str(model)+str(data)).hexdigest()[:10]

class Runner():
    def __init__(self, data, model):
        self.data = data
        self.model = model
        model.logger = self.pandas_logger
        self.logger_file = None #open("logs/{}.log".format(utils.hash(self.model, self.data)), "w")
        self.log = pd.Series(index=self.data.get_dataframe() .index)
        self.hash = get_hash(self.model, self.data)

    def file_logger(self, log):
        self.logger_file.write("{}\n".format(log))

    def pandas_logger(self, answer, prediction):
        self.log[answer["id"]] = prediction

    def clean(self):
        os.remove("cache/{}.report".format(self.hash))
        os.remove("cache/{}.pd".format(self.hash))


    def run(self):
        start = datetime.datetime.now()
        print "Pre-processing data..."
        self.model.pre_process_data(self.data)
        pre_processing_time = datetime.datetime.now() - start
        print pre_processing_time

        start = datetime.datetime.now()
        print "Processing data..."
        self.model.process_data(self.data)
        processing_time = datetime.datetime.now() - start
        print processing_time

        report = {
            "model": str(self.model),
            "data": str(self.data),
            "processing time": str(processing_time),
            "pre-processing time": str(pre_processing_time),
            "data count": self.data.n,
        }

        with open("cache/{}.report".format(self.hash), "w") as f:
            json.dump(report, f)

        self.log.to_pickle("cache/{}.pd".format(self.hash))

        print "Written to {} report and log".format(self.hash)

