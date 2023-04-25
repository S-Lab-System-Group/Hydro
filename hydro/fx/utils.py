from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple, Iterable

import re
import ast
import torch.fx as fx


VALID_AST_TYPES = [int, float, bool, tuple, list, dict]


def get_parent_name(target: str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit(".", 1)
    return parent[0] if parent else "", name


# Parse string to kwargs
def parse_param_dict(extra_lines: List, kwargs: Dict):
    for i in range(0, len(extra_lines), 2):
        value = ast.literal_eval(extra_lines[i + 1])
        if type(value) in VALID_AST_TYPES:
            kwargs.update({extra_lines[i]: value})
        else:
            raise NotImplementedError
    return kwargs


def parse_params_repr(repr: str):
    args, kwargs = [], {}

    repr = repr.replace(" ", "")
    extra_lines = re.split(r",|=", repr)

    for i in range(len(extra_lines) - 1):
        if "(" in extra_lines[i] and ")" in extra_lines[i + 1]:
            ori_tuple_str = ",".join([extra_lines[i], extra_lines[i + 1]])
            extra_lines[i] = ori_tuple_str
            extra_lines[i + 1] = ""

    extra_lines = [i for i in extra_lines if i != ""]

    # Parse string to args
    for i, _ in enumerate(extra_lines):
        try:
            arg = ast.literal_eval(extra_lines[i])
            if type(arg) in VALID_AST_TYPES:
                args.append(arg)
            else:
                raise NotImplementedError
        except ValueError:
            kwargs = parse_param_dict(extra_lines[i:], kwargs)
            break

    return args, kwargs


def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    if node.op != "call_module":
        return False
    if not isinstance(node.target, str):
        return False
    if node.target not in modules:
        return False
    for ops in pattern:
        if type(modules[node.target]) is ops:
            return True
    return False
