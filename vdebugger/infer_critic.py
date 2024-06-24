#!/usr/bin/env python
# coding=utf-8
import argparse
import ast
import os

import pandas as pd
import sklearn.metrics
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from my_datasets import datasets, process_result


def parse(code):
    try:
        return ast.unparse(ast.parse(code))
    except:
        return None


@torch.no_grad()
def main(args):
    if os.path.exists(args.model):
        output_fname = os.path.join(args.model, args.output_fname)
    assert not os.path.exists(output_fname)
    print("Dump to", output_fname)

    tokenizer = AutoTokenizer.from_pretrained('codellama/CodeLlama-7b-Python-hf')
    model = LLM(args.model, tokenizer='codellama/CodeLlama-7b-Python-hf',
                tensor_parallel_size=torch.cuda.device_count())

    sampling_params = SamplingParams(temperature=0, max_tokens=256)

    dataset = datasets[args.dataset]
    data = pd.read_csv(args.input)
    input_ids = []
    labels = []
    for i, row in data.iterrows():
        q1, q2 = row['query'].splitlines()
        code = row['code']
        if not code.startswith(q2):
            code = q2 + code
        code_ = parse(code)
        if code_ is not None:
            code = code_

        custom_trace = row['traced']
        if pd.isna(custom_trace):
            error = row['error']
            assert 'CompileTimeError' in error
            custom_trace = 'Compile Error'
        info = "\n\n-> {}\n\n--- Trace\n\n{}".format(row['result'], custom_trace)

        prompt = "# {}\n{}{}".format(q1, code, info)

        tokenized_prompt = [tokenizer.bos_token_id, ] + tokenizer(prompt, add_special_tokens=False).input_ids
        tokenized_inst = tokenizer('\n\n# Program is', add_special_tokens=False).input_ids
        tokenized_prompt = tokenized_prompt[: 1024 - 256 - len(tokenized_inst)]
        input_ids.append(tokenized_prompt + tokenized_inst)
        labels.append(int(dataset.accuracy([process_result(row['result']), ], [row['answer'], ])))

    generation = model.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    generation = [g.outputs[0].text for g in generation]
    preds = [x.strip().startswith("right") for x in generation]

    print("acc = {}".format(sklearn.metrics.accuracy_score(labels, preds)))
    print("confusion matrix:")
    print(sklearn.metrics.confusion_matrix(labels, preds))

    data.insert(len(data.keys()), 'critic', '')
    data['critic'] = generation
    data.to_csv(output_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output-fname', default='critic-infer.csv')
    parser.add_argument('--dataset', default='gqa')
    args = parser.parse_args()

    main(args)
