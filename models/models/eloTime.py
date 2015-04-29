from collections import defaultdict
from model import Model, sigmoid


class EloTimeModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, decay_function=None, time_penalty_slope=0.8):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)

        self.global_skill = defaultdict(lambda: 0)
        self.difficulty = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.time_penalty_slope = time_penalty_slope

    def __str__(self):
        return "Elo; decay - alpha: {}, beta: {}, slope: {}".format(self.alpha, self.beta, self.time_penalty_slope)

    def predict(self, student, item, extra=None):
        random_factor = 0 if extra is None or extra.get("choices", 0) == 0 else 1. / extra["choices"]
        prediction = sigmoid(self.global_skill[student] - self.difficulty[item], random_factor)
        return prediction

    def update(self, student, item, prediction, correct, extra=None):
        response = self._get_response(correct, extra["response_time"])
        dif = (response - prediction)

        self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
        self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif
        self.student_attempts[student] += 1
        self.item_attempts[item] += 1

    def _get_response(self, correct, response_time):
        if not correct:
            return 0

        expected_solving_time = 5
        if expected_solving_time > response_time:
            return 1

        return self.time_penalty_slope ** ((response_time / expected_solving_time) - 1)
