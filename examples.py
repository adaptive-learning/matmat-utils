from answers import *
from skills import *
import pylab as plt


def response_time_hist(answers=None):
    answers = answers if answers is not None else load_answers()
    answers = filter_long_times(answers)

    answers["solving_time"].hist(bins=50, range=(0, 50))
    plt.show()


def time_accuracy_trade_off(answers=None):
    answers = answers if answers is not None else load_answers()

    points = range(0, 51)
    counts = []
    success_rate = []

    for i in points:
        ans = answers[(i <= answers["solving_time"]) & (answers["solving_time"] < i+1)]
        counts.append(len(ans))
        success_rate.append(ans["correctly_solved"].mean())

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(points, counts)
    ax1.set_ylabel("answers")
    ax1.set_xlabel("response time")
    ax2.plot(points, success_rate, "g")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("success rate")

    plt.show()

time_accuracy_trade_off()