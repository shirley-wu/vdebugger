import os
import signal
import traceback
import warnings
from functools import partial

import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from main_utils import save_results_csv, my_collate, TimeOutException
from utils import seed_everything

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)


def handler(signum, frame):
    print("Code execution timeout")
    raise TimeOutException()


def run_program(parameters, queues_in_, input_type_, retrying=False):
    from image_patch import ImagePatch, llm_query, best_image_match, distance, bool_to_yesno
    from video_segment import VideoSegment

    global queue_results

    code, sample_id, image, possible_answers, query = parameters
    code = str(code)
    assert 'def ' not in str(query)

    if code.startswith("\ndef"):
        code = code[1:]  # TODO: just a temporary fix

    code_header = f'def execute_command_{sample_id}(' \
                  f'{input_type_}, possible_answers, query, ' \
                  f'ImagePatch, VideoSegment, ' \
                  'llm_query, bool_to_yesno, distance, best_image_match):\n' \
                  f'    # Answer is:'
    if not isinstance(code, str):
        print("Warning! code:")
        print(code)
        code = str(code)
    code = code_header + code

    try:
        exec(compile(code, 'Codex', 'exec'), globals())
    except Exception as e:
        print(f'Sample {sample_id} failed at compilation time with error: {e}')
        try:
            with open(config.fixed_code_file, 'r') as f:
                fixed_code = f.read()
            code = code_header + fixed_code
            exec(compile(code, 'Codex', 'exec'), globals())
        except Exception as e2:
            print(f'Not even the fixed code worked. Sample {sample_id} failed at compilation time with error: {e2}')
            return None, code

    queues = [queues_in_, queue_results]

    image_patch_partial = partial(ImagePatch, queues=queues)
    video_segment_partial = partial(VideoSegment, queues=queues)
    llm_query_partial = partial(llm_query, queues=queues)

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60 * 20)  # timeout = 10min, just in case while True
        result = globals()[f'execute_command_{sample_id}'](
            # Inputs to the function
            image, possible_answers, query,
            # Classes to be used
            image_patch_partial, video_segment_partial,
            # Functions to be used
            llm_query_partial, bool_to_yesno, distance, best_image_match)
    except Exception as e:
        # print full traceback
        traceback.print_exc()
        if retrying:
            return None, code
        print(f'Sample {sample_id} failed with error: {e}. Next you will see an "expected an indented block" error. ')
        # Retry again with fixed code
        new_code = "["  # This code will break upon execution, and it will be caught by the except clause
        result = run_program((new_code, sample_id, image, possible_answers, query), queues_in_, input_type_,
                             retrying=True)[0]
    finally:
        signal.alarm(0)

    # The function run_{sample_id} is defined globally (exec doesn't work locally). A cleaner alternative would be to
    # save it in a global dict (replace globals() for dict_name in exec), but then it doesn't detect the imported
    # libraries for some reason. Because defining it globally is not ideal, we just delete it after running it.
    if f'execute_command_{sample_id}' in globals():
        del globals()[f'execute_command_{sample_id}']  # If it failed to compile the code, it won't be defined
    return result, code


def worker_init(queue_results_):
    global queue_results
    index_queue = mp.current_process()._identity[0] % len(queue_results_)
    queue_results = queue_results_[index_queue]


def main():
    print(config)

    mp.set_start_method('spawn')

    from vision_processes import queues_in, finish_all_consumers, forward, manager
    from datasets import get_dataset

    batch_size = config.dataset.batch_size
    num_processes = min(batch_size, 50)

    if config.multiprocessing:
        queue_results_main = manager.Queue()
        queues_results = [manager.Queue() for _ in range(batch_size)]
    else:
        queue_results_main = None
        queues_results = [None for _ in range(batch_size)]

    model_name_codex = 'codellama' if config.codex.model == 'codellama' else 'codex'
    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    assert config.execute_code
    dataset = get_dataset(config.dataset, load_image=True, orig_query=True)

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()

    codes_all = None
    if config.use_cached_codex:
        results = pd.read_csv(config.cached_codex_path)
        # codes_all = [r.split('# Answer is:')[1] for r in results['code']]
        codes_all = results['code'].tolist()
    # python -c "from joblib import Memory; cache = Memory('cache/', verbose=0); cache.clear()"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True,
                            collate_fn=my_collate)
    input_type = dataset.input_type

    all_results = []
    all_answers = []
    all_codes = []
    all_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []

    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=n_batches):

                # Combine all queries and get Codex predictions for them
                # TODO compute Codex for next batch as current batch is being processed

                if not config.use_cached_codex:
                    raise NotImplementedError("Please only use this script for executing visual programs")

                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache

                # Run the code
                if config.execute_code:
                    if not config.multiprocessing:
                        # Otherwise, we would create a new model for every process
                        results = []
                        for c, sample_id, img, possible_answers, query in \
                                zip(codes, batch['sample_id'], batch['img'], batch['possible_answers'],
                                    batch['question']):
                            result = run_program([c, sample_id, img, possible_answers, query], queues_in, input_type)
                            results.append(result)
                    else:
                        results = list(pool.imap(partial(
                            run_program, queues_in_=queues_in, input_type_=input_type),
                            zip(codes, batch['sample_id'], batch['img'], batch['possible_answers'], batch['question'])))
                else:
                    results = [(None, c) for c in codes]
                    warnings.warn("Not executing code! This is only generating the code. We set the flag "
                                  "'execute_code' to False by default, because executing code generated by a language "
                                  "model can be dangerous. Set the flag 'execute_code' to True if you want to execute "
                                  "it.")

                all_results += [r[0] for r in results]
                all_codes += [r[1] for r in results]
                all_ids += batch['sample_id']
                all_answers += batch['answer']
                all_possible_answers += batch['possible_answers']
                all_query_types += batch['question_type']
                all_queries += batch['question']
                all_img_paths += ['' for idx in batch['index']]
                if i % config.log_every == 0:
                    try:
                        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types,
                                                    strict=False)
                        console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                    except Exception as e:
                        console.print(f'Error computing accuracy: {e}')

        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    try:
        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types, strict=False)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    if config.save:
        df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_queries, all_img_paths,
                           all_possible_answers]).T
        df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers']
        # make the result column a string
        df['result'] = df['result'].apply(str)
        # df['error'] = df['error'].apply(str)
        filename = save_results_csv(config, df)
        config['py_file'] = __file__
        OmegaConf.save(config=config, f=filename.replace(".csv", ".yaml"))
        print("Dump finished")

        assert not config.wandb

    finish_all_consumers()


if __name__ == '__main__':
    main()
