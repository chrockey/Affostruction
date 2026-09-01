from typing import *

BACKEND = 'flash_attn'
DEBUG = False

def __from_env():
    import os

    global BACKEND
    global DEBUG

    env_attn_backend = os.environ.get('ATTN_BACKEND')
    env_sttn_debug = os.environ.get('ATTN_DEBUG')

    if env_attn_backend is not None:
        env_attn_backend = env_attn_backend.replace('-', '_')
    if env_attn_backend is not None and env_attn_backend in ['xformers', 'flash_attn', 'sdpa', 'naive']:
        BACKEND = env_attn_backend
    if env_sttn_debug is not None:
        DEBUG = env_sttn_debug == '1'

    if BACKEND == 'flash_attn':
        import importlib.util
        if importlib.util.find_spec('flash_attn') is None:
            print('[ATTENTION] flash_attn not installed; falling back to sdpa')
            BACKEND = 'sdpa'

    print(f"[ATTENTION] Using backend: {BACKEND}")


__from_env()


def set_debug(debug: bool):
    global DEBUG
    DEBUG = debug


from .full_attn import *
from .modules import *
