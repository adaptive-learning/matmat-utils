from data.data import Data
from models.model import AvgModel, AvgItemModel
import pylab as plt
import seaborn as sbn
from models.elo import EloModel
from models.eloTree import EloTreeModel
from models.eloCurrent import EloPriorCurrentModel
from skills import load_skills, load_questions, get_question_parents, get_skill_parents
from utils import elo_grid_search, compare_models, compare_brier_curve, get_skills, elo_pfa_search
import pandas as pd

data = Data("data/matmat-all.pd", train=0)
qp, sp = get_question_parents(load_questions()), get_skill_parents(load_skills())

compare_models(data, [
    AvgModel(),
    AvgItemModel(),
    # EloModel(),
    EloModel(alpha=0.8, beta=0),
    EloTreeModel(qp, sp, alpha=1.2, beta=0.1, KC=3.5, KI=2.5),
    # EloPriorCurrentModel(alpha=0.8, beta=0, KC=1, KI=1),
    EloPriorCurrentModel(alpha=0.8, beta=0, KC=2.5, KI=1),
], dont=0, evaluate=0)

# compare_brier_curve(data, AvgItemModel(), EloModel(alpha=0.8, beta=0))
# elo_grid_search(data, beta_range=(0, 0.1, 0.02), model_class=EloModel)
# elo_grid_search(data, beta_range=(0, 0.1, 0.02), model_class=EloPriorCurrentModel)
# elo_grid_search(data, beta_range=(0, 0.2, 0.02), model_class=EloTreeModel)

# elo_pfa_search(data, model_class=EloPriorCurrentModel)
# elo_pfa_search(data, model_class=EloTreeModel)

# pd.Series(get_skills(data, EloModel(alpha=0.8, beta=0)), name="skill").to_pickle("../data/skills-Elo.pd")

plt.show()