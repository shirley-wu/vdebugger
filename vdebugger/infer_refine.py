#!/usr/bin/env python
# coding=utf-8
import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse(code):
    try:
        return ast.unparse(ast.parse(code))
    except:
        return None


@torch.no_grad()
def main(args):
    if os.path.exists(args.refine):
        output_fname = os.path.join(args.refine, args.output_fname)
    assert not os.path.exists(output_fname)
    print("Dump to", output_fname)

    tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Python-hf')
    model = LLM(args.refine, tokenizer='codellama/CodeLlama-7b-Python-hf',
                tensor_parallel_size=torch.cuda.device_count())

    sampling_params = SamplingParams(temperature=0, max_tokens=256)

    critic_outputs = pd.read_csv(args.critic)
    inferred_inds = []
    input_ids = []
    for i in range(len(critic_outputs)):
        if isinstance(critic_outputs['critic'][i], str) and \
                not critic_outputs['critic'][i].strip().startswith('right'):
            q1, q2 = critic_outputs['query'][i].splitlines()
            try:
                code = 'def execute_command' + critic_outputs['critic'][i].strip().split('def execute_command')[1]
            except:
                print("Invalid critic generation at %d:" % i)
                print(critic_outputs['critic'][i])
                continue

            if '# Program is' in code:
                code = code.split("# Program is")[0].strip()  # errr, an awkward fix
            if args.info == 'trace':
                info = '\n\n-> {}\n\n--- Trace\n\n{}'.format(
                    critic_outputs['result'][i], critic_outputs['traced'][i])
            else:
                info = ''
            prompt = "# {}\n{}{}".format(q1, code, info)
            tokenized_prompt = [tokenizer.bos_token_id, ] + tokenizer(prompt, add_special_tokens=False).input_ids
            tokenized_inst = tokenizer('\n\n# Correction', add_special_tokens=False).input_ids
            tokenized_prompt = tokenized_prompt[: 1024 - 256 - len(tokenized_inst)]
            inferred_inds.append(i)
            input_ids.append(tokenized_prompt + tokenized_inst)

    if args.drop_unchanged:
        critic_outputs['code'] = '['

    generation = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    generation = [g.outputs[0].text.strip() for g in generation]
    critic_outputs.loc[np.array(inferred_inds), "code"] = generation
    critic_outputs.to_csv(output_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('critic')
    parser.add_argument('refine')
    parser.add_argument('--output-fname', default='critic-refine-infer.csv')
    parser.add_argument('--drop-unchanged', default=False, action='store_true')
    args = parser.parse_args()

    main(args)
