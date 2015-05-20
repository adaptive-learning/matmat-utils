from collections import defaultdict
from model import Model, sigmoid
from skills import get_skill_parents, load_skills, get_question_parents


class EloTreeDecayModel(Model):

    def __init__(self, item_parents, skill_parents, alpha=1.0, beta=0.1, KC=3.5, KI=2.5, without_decay=False):
        Model.__init__(self)

        self.alpha = alpha
        self.beta = beta
        self.KC = KC
        self.KI = KI
        self.decay_function = lambda x: alpha / (1 + beta * x)
        self.without_decay = without_decay

        self.skill_parents = skill_parents
        self.item_parents = item_parents
        self.skill = defaultdict(lambda: defaultdict(lambda: 0))
        self.difficulty = defaultdict(lambda: 0)
        self.student_attempts = defaultdict(lambda: defaultdict(lambda: 0))
        self.item_attempts = defaultdict(lambda: 0)
        self.first_attempt = defaultdict(lambda: defaultdict(lambda: True))

    def __str__(self):
        return "Elo Tree Decay; decay - alpha: {}, beta: {}, KC: {}, KI: {}{}".format(self.alpha, self.beta, self.KC, self.KI, " without decay2" if self.without_decay else "")

    def predict(self, student, item, extra=None):
        skill = self._get_skill(student, self.item_parents[item])
        prediction = sigmoid(skill - self.difficulty[item])

        return prediction

    def update(self, student, item, prediction, correct, extra=None):
        dif = (correct - prediction)

        for level, skill in enumerate(self._get_parents(item)):
            p = sigmoid(self._get_skill(student, skill) - self.difficulty[item])
            K = self.KC if correct else self.KI
            decay = 1 if self.without_decay else self.decay_function(self.student_attempts[skill][student])
            self.skill[skill][student] += decay * (correct - p) * K
            self.student_attempts[skill][student] += 1

        if self.first_attempt[item][student]:
            self.difficulty[item] -= self.decay_function(self.item_attempts[item]) * dif

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