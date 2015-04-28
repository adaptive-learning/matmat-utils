from answers import convert_answers_to_model, load_answers
from data.data import Data
from evaluator import compare_models
from models.model import AvgModel, AvgItemModel
import pylab as plt
import seaborn as sbn
from models.elo import EloModel
from utils import elo_grid_search

data = Data("data/matmat-all.pd", train=0.3)


compare_models(data, [
    AvgModel(),
    AvgItemModel(),
    EloModel(),
    EloModel(alpha=0.8, beta=0),
], dont=0, evaluate=0)


# elo_grid_search(data, beta_range=(0, 0.1, 0.02))

plt.show()