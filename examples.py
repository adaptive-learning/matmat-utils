from answers import *
from skills import *
import pylab as plt


def response_time_hist():
    answers = load_answers()
    answers = filter_long_times(answers)

    answers["solving_time"].hist(bins=50, range=(0, 50))
    plt.show()