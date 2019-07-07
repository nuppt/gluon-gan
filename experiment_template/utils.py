import mxnet as mx
from mxnet import nd
import json


def read_expr_config(conf_file: str):
    """
    Read dict from config file  ->  convert to DotMap
    DotMap 允许我们通过 dot access 的方式，访问每一个字段，这样可以和 argparse 对象保持一致。
    :param conf_file:
    :return:
    """
    with open(conf_file) as f:
        json_dict = json.load(f)
    return json_dict


def try_gpu(gpu_id=0):
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu(gpu_id)
        _ = nd.array([0], ctx=ctx)
    except Exception:
        ctx = mx.cpu()
    return ctx


def try_all_gpus():
    """Return all available GPUs, or [mx.gpu()] if there is no GPU"""
    ctx_list = []
    try:
        for i in range(16):
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctx_list.append(ctx)
    except Exception:
        pass
    if not ctx_list:
        ctx_list = [mx.cpu()]
    return ctx_list
