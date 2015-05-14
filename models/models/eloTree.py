from collections import defaultdict
from model import Model, sigmoid
from skills import get_skill_parents, load_skills, get_question_parents


class EloTreeModel(Model):

    def __init__(self, item_parents, skill_parents, alpha=1.0, beta=0.1, KC=1, KI=1, decay_function=None, level_decay=3):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.KC = KC
        self.KI = KI
        self.decay_function = decay_function if decay_function is not None else lambda x: alpha / (1 + beta * x)
        self.level_decay = level_decay
        self.level_decay_fce = lambda level: 1. / level_decay ** (3 - level)

        self.skill_parents = skill_parents
        self.item_parents = item_parents
        self.skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.difficulty = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: 0)
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def __str__(self):
        return "Elo Tree; decay - alpha: {}, beta: {}, KC: {}, KI: {}, LD: {}".format(self.alpha, self.beta, self.KC, self.KI, self.level_decay)

    def predict(self, student, item, extra=None):
        skill = self._get_skill(student, self.item_parents[item])
        prediction = sigmoid(skill - self.difficulty[item])

        return prediction

    def update(self, student, item, prediction, correct, extra=None):
        dif = (correct - prediction)

        for level, skill in enumerate(self._get_parents(item)):
            p = sigmoid(self._get_skill(student, skill) - self.difficulty[item])
            K = self.KC if correct else self.KI
            self.skill[skill][student] += self.level_decay_fce(level) * (correct - p) * K

        if self.first_attempt[item][student]:
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif

        self.student_attempts[student] += 1
        self.item_attempts[item] += 1
        self.first_attempt[item][student] = False

    def _get_skill(self, student, skill):
        skill_value = 0
        while skill is not None:
            skill_value += self.skill[skill][student]
            skill = self.skill_parents[skill]
        return skill_value

    def _get_parents(self, item):
        skills = []
        skill = self.item_parents[item]
        while skill is not None:
            skills.append(skill)
            skill = self.skill_parents[skill]
        return skills[::-1]