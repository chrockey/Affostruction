SPCONV_ALGO = "auto"


def __from_env():
    import os

    global SPCONV_ALGO
    env_spconv_algo = os.environ.get("SPCONV_ALGO")
    if env_spconv_algo is not None and env_spconv_algo in ["auto", "implicit_gemm", "native"]:
        SPCONV_ALGO = env_spconv_algo


__from_env()

from .conv_spconv import *
