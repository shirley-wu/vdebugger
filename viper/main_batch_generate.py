import json
import os
import pickle
import traceback
from functools import partial

import pandas as pd
import torch.multiprocessing as mp
from joblib import Memory
from omegaconf import OmegaConf
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import config
from main_utils import CompileTimeError, ProgramRuntimeError, save_results_csv, my_collate
from utils import seed_everything

# See https://github.com/pytorch/pytorch/issues/11201, https://github.com/pytorch/pytorch/issues/973
# Not for dataloader, but for multiprocessing batches
mp.set_sharing_strategy('file_system')
queue_results = None

cache = Memory('cache/' if config.use_cache else None, verbose=0)
# STAMP = sys.argv[1]
# if os.path.exists(f'cache/{STAMP}'):
#     os.removedirs(f'cache/{STAMP}')
# cache = Memory(f'cache/{STAMP}', verbose=0)
runs_dict = {}
seed_everything()
console = Console(highlight=False)


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

    if config.codex.model == 'codellama':
        if getattr(config.codex, 'multi_turn', False):
            model_name_codex = 'multiturn_codellama'
        else:
            model_name_codex = 'codellama'
    else:
        if getattr(config.codex, 'multi_turn', False):
            model_name_codex = 'multiturn_codex'
        else:
            model_name_codex = 'codex'
    codex = partial(forward, model_name=model_name_codex, queues=[queues_in, queue_results_main])

    if config.clear_cache:
        cache.clear()

    if config.wandb:
        import wandb
        wandb.init(project="viper", config=OmegaConf.to_container(config))
        # log the prompt file
        wandb.save(config.codex.prompt)

    dataset = get_dataset(config.dataset, load_image=config.execute_code)

    with open(config.codex.prompt) as f:
        base_prompt = f.read().strip()
    if getattr(config.codex, 'multi_turn', False):
        base_prompt = base_prompt.split("<<<<SPLIT>>>>")
        base_prompt = [p.strip() for p in base_prompt]

    codes_all = None
    if config.use_cached_codex:
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

    all_results = []
    all_errors = []
    all_answers = []
    all_codes = []
    all_codes_candidates = []
    all_ids = []
    all_queries = []
    all_img_paths = []
    all_possible_answers = []
    all_query_types = []
    filename = None

    with mp.Pool(processes=num_processes, initializer=worker_init, initargs=(queues_results,)) \
            if config.multiprocessing else open(os.devnull, "w") as pool:
        try:
            n_batches = len(dataloader)

            for i, batch in tqdm(enumerate(dataloader), total=n_batches):
                codes_candidates = [[] for _ in range(len(batch['question']))]
                if not config.use_cached_codex:
                    codes = codex(prompt=batch['question'], base_prompt=base_prompt, input_type=input_type,
                                  extra_context=batch.get(
                                      'extra_context', ['' for _ in range(len(batch['question']))]
                                  ))
                    if getattr(config.codex, "overgenerate", False):
                        codes_candidates = [json.dumps(c) for c in codes]
                        codes = [c[0] for c in codes]

                else:
                    codes = codes_all[i * batch_size:(i + 1) * batch_size]  # If cache
                    assert len(codes) == len(batch['answer']), "Cached code breaks"

                if i * len(codes) < 4:
                    print("First %d - %d sample:" % (i * len(codes), (i + 1) * len(codes)))
                    for c in codes:
                        print(c)
                        print()

                # Run the code
                assert not config.execute_code, "Please only use this script for generating visual programs"
                results = [(None, c) for c in codes]

                all_results += [r[0] for r in results]
                all_errors += [r[-1] for r in results]
                all_codes += [r[1] for r in results]
                all_codes_candidates += codes_candidates
                all_ids += batch['sample_id']
                all_answers += batch['answer']
                all_possible_answers += batch['possible_answers']
                all_query_types += batch['question_type']
                all_queries += batch['question']
                # all_img_paths += [dataset.get_sample_path(idx) for idx in batch['index']]
                all_img_paths += ['' for idx in batch['index']]
                if i % config.log_every == 0:
                    try:
                        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
                        console.print(f'Accuracy at Batch {i}/{n_batches}: {accuracy}')
                    except Exception as e:
                        console.print(f'Error computing accuracy: {e}')

                    if config.save:
                        df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_queries, all_img_paths,
                                           all_possible_answers, all_codes_candidates, all_errors]).T
                        df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers',
                                      'code_candidates',
                                      'error', ]
                        # make the result column a string
                        df['result'] = df['result'].apply(str)
                        df['error'] = df['error'].apply(str)
                        filename = save_results_csv(config, df, filename)
                        config['py_file'] = __file__
                        OmegaConf.save(config=config, f=filename.replace(".csv", ".yaml"))
                        print("Dump finished")


        except Exception as e:
            # print full stack trace
            traceback.print_exc()
            console.print(f'Exception: {e}')
            console.print("Completing logging and exiting...")

    n = sum([isinstance(e, CompileTimeError) for e in all_errors])
    print("%.2f%% (%d out of %d) code execution has compile error" % (n / len(all_answers) * 100, n, len(all_answers)))
    n = sum([isinstance(e, ProgramRuntimeError) for e in all_errors])
    print("%.2f%% (%d out of %d) code execution has runtime error" % (n / len(all_answers) * 100, n, len(all_answers)))

    try:
        accuracy = dataset.accuracy(all_results, all_answers, all_possible_answers, all_query_types)
        console.print(f'Final accuracy: {accuracy}')
    except Exception as e:
        print(f'Error computing accuracy: {e}')

    finish_all_consumers()

    if config.save:
        df = pd.DataFrame([all_results, all_answers, all_codes, all_ids, all_queries, all_img_paths,
                           all_possible_answers, all_codes_candidates, all_errors]).T
        df.columns = ['result', 'answer', 'code', 'id', 'query', 'img_path', 'possible_answers', 'code_candidates',
                      'error', ]
        # make the result column a string
        df['result'] = df['result'].apply(str)
        df['error'] = df['error'].apply(str)
        filename = save_results_csv(config, df, filename)
        config['py_file'] = __file__
        OmegaConf.save(config=config, f=filename.replace(".csv", ".yaml"))
        print("Dump finished")

        assert not config.wandb


if __name__ == '__main__':
    main()
