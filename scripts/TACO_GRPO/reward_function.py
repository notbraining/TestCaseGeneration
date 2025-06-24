import io
import multiprocessing
import time
import builtins
import sys
import resource
import json


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    loaded_ground = json.loads(ground_truth)
    case_input = loaded_ground["inputs"]
    case_output = loaded_ground["outputs"]
    return test_code(solution_str, case_input, case_output, 100)


def run_code(code, case_input, output_queue, memory_limit=256):
    start = time.time()
    code = code.replace("python", "")
    code = code.replace("```", "")
    if isinstance(case_input, list):
        case_input = " ".join(str(s) for s in case_input)

    # Mock input
    input_lines = iter(case_input.splitlines())
    builtins.input = lambda: next(input_lines)

    # Capture output
    buf = io.StringIO()
    sys.stdout = buf
    try:
        exec(code)
    except Exception as _:
        output_queue.put(("RUNTIME ERROR\n", 0))
        return
    finally:
        sys.stdout = sys.__stdout__

    output_queue.put(
        (
            {
                "output": buf.getvalue(),
                "time": time.time() - start,
                "memory": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            },
            0,
        )
    )  # memory will be added later


def test_code(code, cases, ex_out, time_limit):
    score = 0
    correct_cases = 0

    for i in range(len(cases)):
        output_queue = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=run_code, args=(code, cases[i], output_queue)
        )
        p.start()
        p.join(timeout=time_limit)
        if p.is_alive():
            p.terminate()
            out = "TIME LIMIT EXCEEDED\n"
            score -= 1 / len(cases) * 10
        else:
            try:
                out, _ = output_queue.get(timeout=1)
            except:
                out = "ERROR\n"

            if out == "ERROR\n" or out == "RUNTIME ERROR\n":
                score -= 1 / len(cases) * 50

        if isinstance(out, dict) and out["output"].rstrip() == ex_out[i].rstrip():
            correct_cases += 1
        else:
            print(out)
    score += correct_cases / len(cases) * 100
    if correct_cases == len(cases):
        score += 25
    return score
