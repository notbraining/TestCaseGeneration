import io
import multiprocessing
from os import wait
import time
import builtins
import sys
import resource
import json
import numpy as n


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    print(solution_str)
    print(ground_truth)
    return 1
    case_input = ground_truth["inputs"]
    case_output = ground_truth["outputs"]
    return eval_solution(solution_str, case_input, case_output)


def eval_solution(solution_str, case_input, case_output):
    reward = 0
    solution_str = re.findall(r"<think>.*</think>(.*)", solution_str, re.DOTALL)
    if solution_str == case_output:
        reward = 1
    return reward
