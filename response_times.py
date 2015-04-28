from answers import *
from skills import *
import pylab as plt
import seaborn as sns
import numpy as np

def response_time_hist(answers=None):
    answers = answers if answers is not None else load_answers()
    answers = filter_long_times(answers)

    sns.set(style="white", palette="muted")
    sns.kdeplot(answers["solving_time"], shade=True, label="answers {}".format(len(answers)))
    # sns.kdeplot(answers.groupby("question")["solving_time"].mean(), shade=True, label="questions {}".format(len(answers["question"].unique())))
    # sns.kdeplot(answers.groupby("user")["solving_time"].mean(), shade=True, label="user {}".format(len(answers["user"].unique())))
    sns.kdeplot(np.exp(answers.groupby("question")["log_times"].mean()), shade=True, label="questions {}".format(len(answers["question"].unique())))
    sns.kdeplot(np.exp(answers.groupby("user")["log_times"].mean()), shade=True, label="user {}".format(len(answers["user"].unique())))
    plt.xlim(0, 30)


def time_accuracy_trade_off(answers=None):
    answers = answers if answers is not None else load_answers()
    answers = filter_long_times(answers)

    points = range(0, 51)
    counts = []
    success_rate = []

    for i in points:
        ans = answers[(i <= answers["solving_time"]) & (answers["solving_time"] < i+1)]
        counts.append(len(ans))
        success_rate.append(ans["correctly_solved"].mean())

    sns.set(style="white", palette="muted")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(points, counts)
    ax1.set_ylabel("answers")
    ax1.set_xlabel("response time")
    ax2.plot(points, success_rate, "g")
    ax2.set_ylim([0, 1])
    ax2.set_ylabel("success rate")


def normed_time_accuracy_trade_off(answers=None, normed_according_to="user"):
    answers = answers if answers is not None else load_answers()
    answers = filter_long_times(answers)
    answers = answers.join(np.exp(answers.groupby(normed_according_to)["log_times"].mean()), on=normed_according_to, rsuffix='_mean')
    answers["normed_time"] = answers["solving_time"] - answers["log_times_mean"]

    points = range(-14, 40)
    counts = []
    shifted_success_rate = []

    for i in points:
        ans = answers[(i <= answers["normed_time"]) & (answers["normed_time"] < i+1)]
        counts.append(len(ans))
        shifted_success_rate.append(ans["correctly_solved"].mean())

    plt.title("Normed response time according to log-mean of {}".format(normed_according_to))
    ax1 = plt.subplot()
    ax1.plot(points, shifted_success_rate, label="success rate")

    ax1.legend(loc=2)
    ax2 = ax1.twinx()
    ax2.plot(points, np.array(counts), color="g", label="number of answers")
    ax2.legend(loc=1)


def log_mean_time_hist(answers=None, normed_according_to="user"):
    answers = answers if answers is not None else load_answers()
    answers = filter_long_times(answers)
    plt.title("Histogram of log-means of response time of {}".format(normed_according_to))
    data = np.exp(answers.groupby(normed_according_to)["log_times"].mean()).to_dict().values()
    sns.distplot(data, color="r", label="log-mean times")
    plt.xlim(0, 30)



answers = get_geography_answers()
# answers = load_answers()

# response_time_hist(answers)
# time_accuracy_trade_off(answers)
# normed_time_accuracy_trade_off(answers)
# normed_time_accuracy_trade_off(answers, normed_according_to="question")
# log_mean_time_hist(answers)
# log_mean_time_hist(answers, normed_according_to="question")

plt.show()

