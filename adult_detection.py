from answers import *
from pylab import plt
import seaborn as snb

children = [39, 292, 293, 294, 523, 524, 618, 1042, 1045, 1048, 1724, 1904]

def speed_vs_success_rate(answers):
    answers = join_response_time_means(answers)
    answers = join_success_rates(answers)
    answers = join_answers_count(answers)
    users = answers.groupby("user").last()

    max = users["user_answer_count"].max() * 1.
    for id, x, y, c, device in zip(users.index, users["user_rt_mean"], users["user_success_rate"], users["user_answer_count"], users["device"]):
        if id not in children:
            plt.plot(x, y, "ok", alpha=c / max)
        else:
            plt.plot(x, y, "or", alpha=c / max)
        if device == "tablet":
            plt.plot(x, y, ".k", alpha=0.5)

    plt.xlabel("Log-mean of response time")
    plt.ylabel("Success rate")
    plt.title("Users: {}-{} answers".format(10, int(max)))


answers = filter_long_times(filter_users(load_answers()))
speed_vs_success_rate(answers)

plt.show()
