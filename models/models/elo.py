from model import Model, sigmoid


class EloModel(Model):

    def __init__(self, alpha=1.0, beta=0.1, decay_function=None, gamma=None, random_factor=None):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if gamma is None:
            self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)
        else:
            self.decay_function = decay_function if decay_function is not None else lambda x: gamma + alpha / (1 + beta * x)

        self.global_skill = {}
        self.difficulty = {}
        self.student_attempts = {}
        self.place_attempts = {}
        self.random_factor = random_factor

    def __str__(self):
        if self.gamma is None:
            return "Elo; decay - alpha: {}, beta: {}{}".format(self.alpha, self.beta, "" if self.random_factor is None else "RF:" + self.random_factor )
        else:
            return "Elo; decay - alpha: {}, beta: {}, gamma: {}".format(self.alpha, self.beta, self.gamma)

    def initialize_if_needed(self, student, item):
        if not student in self.global_skill:
            self.global_skill[student] = 0
            self.student_attempts[student] = 0
        if not item in self.difficulty:
            self.difficulty[item] = 0
            self.place_attempts[item] = 0


    def predict(self, student, item, extra=None):
        self.initialize_if_needed(student, item)
        random_factor = 0 if extra is None or extra.get("choices", 0) == 0 else 1. / extra["choices"]
        prediction = sigmoid(self.global_skill[student] - self.difficulty[item], random_factor)
        return prediction

    def update(self, student, item, prediction, correct, extra=None):
        dif = (correct - prediction)

        self.global_skill[student] += self.decay_function(self.student_attempts[student]) * dif
        self.difficulty[item] -= self.decay_function(self.place_attempts[item]) * dif
        self.student_attempts[student] += 1
        self.place_attempts[item] += 1