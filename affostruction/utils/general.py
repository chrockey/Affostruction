import re
import numpy as np
import cv2
import torch
import contextlib


def dict_foreach(dic, func, special_func={}):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), "input must be a dictionary"
    for key in dic.keys():
        if isinstance(dic[key], dict):
            dic[key] = dict_foreach(dic[key], func)
        else:
            if key in special_func.keys():
                dic[key] = special_func[key](dic[key])
            else:
                dic[key] = func(dic[key])
    return dic


def dict_reduce(dicts, func, special_func={}):
    """
    Reduce a list of dictionaries. Leaf values must be scalars.
    """
    assert isinstance(dicts, list), "input must be a list of dictionaries"
    assert all([isinstance(d, dict) for d in dicts]), "input must be a list of dictionaries"
    assert len(dicts) > 0, "input must be a non-empty list of dictionaries"
    all_keys = sorted({key for dict_ in dicts for key in dict_.keys()})
    reduced_dict = {}
    for key in all_keys:
        vlist = [dict_[key] for dict_ in dicts if key in dict_.keys()]
        if isinstance(vlist[0], dict):
            reduced_dict[key] = dict_reduce(vlist, func, special_func)
        else:
            if key in special_func.keys():
                reduced_dict[key] = special_func[key](vlist)
            else:
                reduced_dict[key] = func(vlist)
    return reduced_dict


def dict_any(dic, func):
    """
    Recursively apply a function to all non-dictionary leaf values in a dictionary.
    """
    assert isinstance(dic, dict), "input must be a dictionary"
    for key in dic.keys():
        if isinstance(dic[key], dict):
            if dict_any(dic[key], func):
                return True
        else:
            if func(dic[key]):
                return True
    return False


def dict_flatten(dic, sep="."):
    """
    Flatten a nested dictionary into a dictionary with no nested dictionaries.
    """
    assert isinstance(dic, dict), "input must be a dictionary"
    flat_dict = {}
    for key in dic.keys():
        if isinstance(dic[key], dict):
            sub_dict = dict_flatten(dic[key], sep=sep)
            for sub_key in sub_dict.keys():
                flat_dict[str(key) + sep + str(sub_key)] = sub_dict[sub_key]
        else:
            flat_dict[key] = dic[key]
    return flat_dict


@contextlib.contextmanager
def nested_contexts(*contexts):
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx())
        yield


def make_grid(images, nrow=None, ncol=None, aspect_ratio=None):
    num_images = len(images)
    if nrow is None and ncol is None:
        if aspect_ratio is not None:
            nrow = int(np.round(np.sqrt(num_images / aspect_ratio)))
        else:
            nrow = int(np.sqrt(num_images))
        ncol = (num_images + nrow - 1) // nrow
    elif nrow is None and ncol is not None:
        nrow = (num_images + ncol - 1) // ncol
    elif nrow is not None and ncol is None:
        ncol = (num_images + nrow - 1) // nrow
    else:
        assert (
            nrow * ncol >= num_images
        ), "nrow * ncol must be greater than or equal to the number of images"

    if images[0].ndim == 2:
        grid = np.zeros(
            (nrow * images[0].shape[0], ncol * images[0].shape[1]), dtype=images[0].dtype
        )
    else:
        grid = np.zeros(
            (nrow * images[0].shape[0], ncol * images[0].shape[1], images[0].shape[2]),
            dtype=images[0].dtype,
        )
    for i, img in enumerate(images):
        row = i // ncol
        col = i % ncol
        grid[
            row * img.shape[0] : (row + 1) * img.shape[0],
            col * img.shape[1] : (col + 1) * img.shape[1],
        ] = img
    return grid


def notes_on_image(img, notes=None):
    img = np.pad(img, ((0, 32), (0, 0), (0, 0)), "constant", constant_values=0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if notes is not None:
        img = cv2.putText(
            img, notes, (0, img.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1
        )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image_with_notes(img, path, notes=None):
    """
    Save an image with notes.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy().transpose(1, 2, 0)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = notes_on_image(img, notes)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def ensure_json_serializable(obj):
    """
    Convert numpy/torch types to JSON-serializable Python types.

    Args:
        obj: Object to convert (dict, list, numpy array, torch tensor, or scalar)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.detach().cpu().numpy().tolist()
    elif hasattr(obj, "item") and not hasattr(obj, "__len__"):
        return obj.item()
    else:
        return obj


def indent(s, n=4):
    """
    Indent a string.
    """
    lines = s.split("\n")
    for i in range(1, len(lines)):
        lines[i] = " " * n + lines[i]
    return "\n".join(lines)
