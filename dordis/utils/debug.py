import logging
import numpy as np
from dordis.config import Config


def _log_sketch(data, round_para, sketch_num, prefix_string, validate=False):
    debug_string = "[Debug]"
    if validate:
        debug_string = "[Validate]"

    first_items = [round(i, round_para) for i in data[:sketch_num]]
    last_items = [round(i, round_para) for i in data[-sketch_num:]]
    ma = round(max(data), round_para)
    mi = round(min(data), round_para)

    logging.info("%s %s First %d/%d are: %s, "
                 "and last %d are: %s. Max: %f, min: %f.",
                 prefix_string, debug_string, sketch_num, len(data),
                 first_items, sketch_num, last_items, ma, mi)


def log_sketch(data, log_prefix_str, mode="server", validate=False):
    debug_params = None
    if hasattr(Config().app, "debug"):
        if mode == "server" and hasattr(Config().app.debug, "server"):
                debug_params = Config().app.debug.server
        elif mode == "client" and hasattr(Config().app.debug, "client"):
                debug_params = Config().app.debug.client

    if debug_params and hasattr(debug_params, "sketch_num"):
        sketch_num = debug_params.sketch_num
        _log_sketch(
            data=data,
            round_para=4,
            sketch_num=sketch_num,
            prefix_string=log_prefix_str,
            validate=validate
        )


def validate_result(res, test_res, prefix_string, involved_clients):
    res = np.array(res).astype(float)
    test_res = np.array(test_res).astype(float)

    diff_norm = np.linalg.norm(res - test_res)
    allclose_res = np.allclose(res, test_res)
    MSE = diff_norm ** 2 / len(res)
    logging.info("%s [Validate] [Sum] Allclose result: %s, "
                 "norm of difference: %.6f, "
                 "MSE: %f.", prefix_string, allclose_res,
                 diff_norm, MSE)

    mean_res = res / len(involved_clients)
    mean_test_res = test_res / len(involved_clients)
    mean_diff_norm = np.linalg.norm(mean_res - mean_test_res)
    mean_allclose_res = np.allclose(mean_res, mean_test_res)
    mean_MSE = mean_diff_norm ** 2 / len(res)
    logging.info("%s [Validate] [Mean] Allclose result: %s, "
                 "norm of difference: %.6f, "
                 "MSE: %f.", prefix_string, mean_allclose_res,
                 mean_diff_norm, mean_MSE)
