import argparse
import ast
import os
import random

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

from my_datasets import datasets


def get_all_candidates(code, head: str):
    code = head + code
    try:
        code = ast.unparse(ast.parse(code))
    except:
        return None

    lines = code.splitlines()
    tree = ast.parse(code)

    def get_text(nodes):
        start = nodes[0]
        end = nodes[-1]
        end_lineno = getattr(end, 'end_lineno', end.lineno)
        if start.lineno == end_lineno:
            return lines[start.lineno - 1][start.col_offset: end.end_col_offset]
        else:
            return '\n'.join(
                [lines[start.lineno - 1][start.col_offset:], ] + lines[start.lineno: end_lineno - 1] + \
                [lines[end_lineno - 1][:end.end_col_offset], ]
            )

    def mask_nodes(nodes):
        start = nodes[0]
        end = nodes[-1]
        ret = lines[:start.lineno - 1] + [lines[start.lineno - 1][:start.col_offset] + '<MASKED>', ]
        ret[-1] += lines[end.end_lineno - 1][end.end_col_offset:]
        ret += lines[end.end_lineno:]
        return '\n'.join(ret)

    def is_ImagePatch(x):
        return isinstance(x, ast.Call) and get_text([x.func, ]) == 'ImagePatch'

    candidate_nodes_to_mask = []
    for node in ast.walk(tree):
        if node is tree:
            continue  # don't add root-level nodes
        if hasattr(node, 'body') and isinstance(node.body, list):
            for i in range(len(node.body)):
                for j in range(i + 1, min(i + 4, len(node.body) + 1)):
                    if not any((isinstance(x, ast.Assign) and is_ImagePatch(x.value)) for x in node.body[i:j]):
                        candidate_nodes_to_mask.append(node.body[i:j])
        if isinstance(node, ast.Assign) or isinstance(node, ast.Return):
            if not is_ImagePatch(node.value) and node.value is not None:
                candidate_nodes_to_mask.append([node.value, ])
        if isinstance(node, ast.If):
            candidate_nodes_to_mask.append([node.test, ])
        if isinstance(node, ast.Call) and is_ImagePatch(node):
            if not is_ImagePatch(node):
                candidate_nodes_to_mask.append(node.args)

    return [(mask_nodes(nodes), get_text(nodes)) for nodes in candidate_nodes_to_mask]


def sample_one_masked_code(code, head):
    candidates = get_all_candidates(code, head)
    if candidates is None:
        return None
    return random.choice(candidates)[0]


def get_prompt():
    with open(os.path.join(os.path.dirname(__file__), '../viper/prompts/benchmarks/joint.py')) as f:
        base_prompt = f.read().strip()
    api_definition = base_prompt.replace("# INSERT_QUERY_HERE", "").strip()
    return """[INST] I am writing code to handle visual question answering tasks by calling computer vision APIs. Some content from the code is masked (represented as "<MASKED>". Please recover the original code.

My code:
```python
# {QUESTION_1}
{CODE}
```

Your code should be wrapped in ```python and ```. The code should be exactly the same as my code, except recovering the masked content.

---

Below are the available APIs and some example usages:

```python
""" + api_definition + """
```[/INST] Here's the original code with the `<MASKED>` section replaced:
```python
# {QUESTION_1}
{QUESTION_2}"""


def main(args):
    dataset = datasets[args.dataset]

    data = pd.read_csv(args.input)
    data['acc'] = [dataset.accuracy([r, ], [a, ]) for r, a in zip(data['result'], data['answer'])]

    PROMPT = get_prompt()
    llm = LLM(model=args.model_id, dtype='bfloat16', tensor_parallel_size=torch.cuda.device_count())
    if args.error_injection == 'mask-best':
        llm.llm_engine.model_executor.driver_worker.model_runner.model.sampler = ErrorSampler()

    data['orig_code'] = data.pop('code')
    data.insert(len(data.keys()), 'masked', '')
    data.insert(len(data.keys()), 'generation', '')
    data.insert(len(data.keys()), 'code', '[')

    prompt_inds = []
    prompts = []
    masked_codes = []
    for i, row in data.iterrows():
        if row['acc'] > 0:
            question_1, question_2 = row['query'].splitlines()
            masked = sample_one_masked_code(row['orig_code'], question_2)
            if masked is not None:
                prompt_inds.append(i)
                masked_codes.append(masked)
                prompt = PROMPT.format(QUESTION_1=question_1, QUESTION_2=question_2, CODE=masked)
                prompts.append(prompt)

    prompt_inds = np.array(prompt_inds)
    data['masked'][prompt_inds] = np.array(masked_codes)

    if args.error_injection == 'mask-best':
        sampling_params = SamplingParams(max_tokens=512, temperature=1.0)
    else:
        assert args.error_injection == 'greedy'
        sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    generation = [o.outputs[0].text for o in outputs]
    new_code = [x.split("```")[0] for x in generation]

    data['generation'][prompt_inds] = np.array(generation)
    data['code'][prompt_inds] = np.array(new_code)

    data.to_csv(args.output, escapechar='\\')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('--model_id', default='codellama/CodeLlama-7b-Instruct-hf')
    parser.add_argument('--error_injection', default='mask-best', choices=['mask-best', 'greedy', ])
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input.replace('.csv', '.error-injection-by-{}.csv'.format(args.error_injection))
    assert not os.path.exists(args.output), "Warning: will overwrite " + args.output

    main(args)
