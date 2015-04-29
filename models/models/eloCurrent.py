from collections import defaultdict
from model import Model, sigmoid


class EloPriorCurrentModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, KC=1, KI=1, decay_function=None):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.KC = KC
        self.KI = KI
        self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)

        self.global_skill = defaultdict(lambda: 0)
        self.local_skill = defaultdict(lambda: defaultdict(lambda: None))
        self.difficulty = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def __str__(self):
        return "EloPriorCurrentModel; decay - alpha: {}, beta: {}, KC: {}, KI: {}".format(self.alpha, self.beta, self.KC, self.KI)

    def predict(self, student, item, extra=None):
        if self.local_skill[item][student] is None:
            self.local_skill[item][student] = self.global_skill[student]

        random_factor = 0 if extra is None or extra.get("choices", 0) == 0 else 1. / extra["choices"]
        prediction = sigmoid(self.local_skill[item][student] - self.difficulty[item], random_factor)
        return prediction

    def update(self, student, item, prediction, correct, extra=None):
        dif = (correct - prediction)

        if self.first_attempt[item][student]:
            self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif
        K = self.KC if correct else self.KI
        self.local_skill[item][student] += K * dif
        self.student_attempts[student] += 1
        self.item_attempts[item] += 1

        self.first_attempt[item][student] = False