import ast
import importlib
import io
import json
import os
import pickle
import re
import signal
import string
import sys
import time
from functools import partial
from typing import List

import pandas as pd
import pysnooper
import torch.multiprocessing as mp
from joblib import Memory
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from main_utils import CompileTimeError, ProgramRuntimeError, TimeOutException, save_results_csv, my_collate
from utils import seed_everything

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)

FUNCTION_HEAD = "def execute_command({input_type}) -> {output_type}:"
EXEC_FUNCTION_HEAD = 'def execute_command({input_type}, possible_answers, query, ImagePatch, VideoSegment,' \
                     ' llm_query, bool_to_yesno, distance, best_image_match):'

STAMP = sys.argv[1]
print("STAMP =", STAMP)


def process_trace(text, function_head, execution_function_head):
    def remove_indent(lines):
        n_space = 0
        for i, c in enumerate(lines[0]):
            if c == ' ':
                n_space += 1
            else:
                break
        return [line[n_space:] if line[0] == ' ' else line for line in lines]

    def remove_pre_context(lines: List[str]):  # lol, just a random use of List
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if execution_function_head in line:
                # assert "call" in line  # TODO: further double-check?
                content = [line.replace(execution_function_head, function_head)] + lines[i + 1:]
                if line[0] == ' ':
                    return remove_indent(content)
                else:
                    return content
        return []

    def remove_post_context(lines):
        for i, line in enumerate(lines):
            if line.startswith("Source path:") and line.endswith(__file__):
                return lines[:i]
            elif line.startswith("Elapsed time"):
                return lines[:i]
        return lines

    def remove_timestamp(lines):
        ret = []
        for line in lines:
            if len(line) > 0 and line[0] in string.digits:
                line = line[16:]  # remove timestamp
            ret.append(line)
        return ret

    def remove_tensor(line):
        return re.sub(r"tensor\(\[\[\[.*?\]\]\]\)", "tensor([[[...]]])", line)

    lines = text.splitlines()
    lines = remove_pre_context(lines)
    lines = remove_post_context(lines)
    lines = remove_timestamp(lines)
    lines = [remove_tensor(line) for line in lines]

    return '\n'.join(lines)


def handler(signum, frame):
    print("Code execution timeout")
    raise TimeOutException()


def run_program_with_trace(parameters, queues_in_, input_type_, output_type_):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno  # noqa
    from video_segment import VideoSegment  # noqa

    function_head = FUNCTION_HEAD.format(input_type=input_type_, output_type=output_type_)
    execution_function_head = EXEC_FUNCTION_HEAD.format(input_type=input_type_, output_type=output_type_)

    global queue_results
    code, sample_id, image, possible_answers, query = parameters

    code = str(code)
    if code.startswith("\ndef"):
        code = code[1:]  # TODO: just a temporary fix

    if code.startswith('def'):
        if code.startswith(function_head):
            code = code.replace(function_head, '')
        else:
            print("--- Code with invalid format\n")
            print(code)
    code = execution_function_head + code
    try:
        code = ast.unparse(ast.parse(code))
    except:
        return None, CompileTimeError(), None

    name = f'x_{STAMP}{sample_id}'
    with open(f'{name}.py', 'w') as f:
        f.write(code)

    for _ in range(20):
        try:
            x = importlib.import_module(name)
        except ModuleNotFoundError:
            print("Errrr, import error. Wait a bit while.")
            time.sleep(60)  # I have no idea why it sometimes fails. Probably file system error
        except Exception as e:
            print("Import has error:", e)
            break
        else:
            break

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(60 * 20)  # timeout = 10min, just in case while True
    with io.StringIO() as f:
        with pysnooper.snoop(output=f, color=False, depth=2, max_variable_length=1000):
            result = None
            error = None
            try:
                result = x.execute_command(image, possible_answers, query, image_patch_partial, video_segment_partial,
                                           llm_query_partial, bool_to_yesno, distance, best_image_match)
            except:
                error = ProgramRuntimeError()
            finally:
                signal.alarm(0)
            os.remove(f'{name}.py')
        f.seek(0)
        traced = f.read(100000)
        traced_processed = process_trace(traced, function_head, execution_function_head)

    try:
        _ = pickle.dumps(result)
    except:
        print("Pickle dump fails for {} type object".format(type(result)))
        print("Convert result to str")
        result = str(result)

    return result, error, traced_processed


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


def main():
    print(config)

    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, manager
    from datasets import get_dataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    if config.clear_cache:
        cache.clear()

    dataset = get_dataset(config.dataset, load_image=True)

    assert config.use_cached_codex
    if config.cached_codex_path.endswith(".csv"):
        results = pd.read_csv(config.cached_codex_path)
    else:
        assert config.cached_codex_path.endswith(".pkl")
        with open(config.cached_codex_path, 'rb') as f:
            results = pickle.load(f)
    # codes_all = [r.split('# Answer is:')[1] for r in results['code']]
    codes_all = results['code'].tolist()
    # python -c "from joblib import Memory; cache = Memory('cache/', verbose=0); cache.clear()"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type
    output_type = dataset.output_type

    all_codes = results['code'].tolist()
    if 'code_candidates' in results:
        all_codes_candidates = results['code_candidates'].tolist()
        try:
            all_codes_candidates = [json.loads(x) for x in all_codes_candidates]
        except:
            print("`code_candidates` corrupted, shame")
            all_codes_candidates = [None for x in all_codes_candidates]
    else:
        all_codes_candidates = [[] for _ in all_codes]

    all_results = []
    all_errors = []
    all_results_traced = []
    all_answers = []
    all_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []
    filename = None

    n_batches = len(dataloader)
    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        for i, batch in tqdm(enumerate(dataloader), total=n_batches):
            assert config.use_cached_codex
            codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache
            assert len(codes) == len(batch['answer']), "Cached code breaks"

            if i * len(codes) < 4:
                print("First %d - %d sample:" % (i * len(codes), (i + 1) * len(codes)))
                for c in codes:
                    print(c)
                    print()

            # Run the code
            assert config.execute_code
            if not config.multiprocessing:
                # Otherwise, we would create a new model for every process
                results = []
                for c, sample_id, img, possible_answers, query in zip(
                        codes, batch['sample_id'], batch['img'], batch['possible_answers'], batch['question']):
                    result = run_program_with_trace([c, sample_id, img, possible_answers, query], queues_in,
                                                    input_type, output_type)
                    results.append(result)
            else:
                # Note: not recommended to use multiprocessing
                results = list(pool.imap(partial(
                    run_program_with_trace, queues_in_=queues_in, input_type_=input_type, output_type_=output_type),
                    zip(codes, batch['sample_id'], batch['img'], batch['possible_answers'], batch['question'])))

            if i * len(codes) < 4:
                print("First %d - %d sample, traced:" % (i * len(codes), (i + 1) * len(codes)))
                for _, error, traced in results:
                    print(str(traced)[:4000])
                    print('Error =', error)
                    print()

            all_results += [r[0] for r in results]
            all_errors += [r[1] for r in results]
            all_results_traced += [r[2] for r in results]
            all_ids += batch['sample_id']
            all_answers += batch['answer']
            all_possible_answers += batch['possible_answers']
            all_query_types += batch['question_type']
            all_queries += batch['question']
            all_img_paths += ['' for idx in batch['index']]
            if i % config.log_every == 0:
                try:
                    accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                    console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                except Exception as e:
                    console.print(f'Error computing accuracy: {e}')

                if config.save:
                    df = pd.DataFrame(
                        [all_results, all_answers, all_results_traced, all_errors, all_codes, all_ids, all_queries,
                         all_img_paths, all_possible_answers, all_codes_candidates]).T
                    df.columns = ['result', 'answer', 'traced', 'error', 'code', 'id', 'query', 'img_path',
                                  'possible_answers',
                                  'code_candidates', ]
                    # make the result column a string
                    df['result'] = df['result'].apply(str)
                    df['error'] = df['error'].apply(str)
                    filename = save_results_csv(config, df, filename)
                    config['py_file'] = __file__
                    OmegaConf.save(config=config, f=filename.replace(".csv", ".yaml"))
                    print("Dump finished")

    n_compile_error = sum([isinstance(e, CompileTimeError) for e in all_errors])
    print("%.2f%% (%d out of %d) code execution has compile error" % (
        n_compile_error / len(all_answers) * 100, n_compile_error, len(all_answers),
    ))

    try:
        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    finish_all_consumers()

    if config.save:
        df = pd.DataFrame([all_results, all_answers, all_results_traced, all_errors, all_codes, all_ids, all_queries,
                           all_img_paths, all_possible_answers, all_codes_candidates]).T
        df.columns = ['result', 'answer', 'traced', 'error', 'code', 'id', 'query', 'img_path', 'possible_answers',
                      'code_candidates', ]
        # make the result column a string
        df['result'] = df['result'].apply(str)
        df['error'] = df['error'].apply(str)
        filename = save_results_csv(config, df, filename)
        config['py_file'] = __file__
        OmegaConf.save(config=config, f=filename.replace(".csv", ".yaml"))
        print("Dump finished")


if __name__ == '__main__':
    main()
