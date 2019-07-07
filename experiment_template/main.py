from dotmap import DotMap
from sacred_cfg import *
from pprint import pprint


@ex.config
def config():
    opt = args
    print(opt)


def dot_opt(opt):
    opt = DotMap(opt)
    print(f"Experiment config opt:")
    pprint(opt)
    return opt


@ex.automain
def main(opt):
    opt = dot_opt(opt)
