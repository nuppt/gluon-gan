EXPERIMENT_CONFIG = "./experiments_config/config.json"
EXPERIMENT_NAME = "GAN-Models"

from utils import read_expr_config
args = read_expr_config(EXPERIMENT_CONFIG)

# sacred 的设置
db_url = args['db_url']
db_name = args['db_name']
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment(EXPERIMENT_NAME + "__" + EXPERIMENT_CONFIG)
ex.observers.append(MongoObserver.create(url=db_url, db_name=db_name))
ex.captured_out_filter = apply_backspaces_and_linefeeds
