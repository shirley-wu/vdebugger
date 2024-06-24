# VDebugger

**VDebugger: Harnessing Execution Feedback for Debugging Visual Programs**

[Paper](https://arxiv.org/abs/2406.13444), [Website](https://shirley-wu.github.io/vdebugger/index.html), [Models and Data](https://huggingface.co/VDebugger)

## Outlines

- [Environment Setup](https://github.com/shirley-wu/vdebugger/tree/main?tab=readme-ov-file#environment-setup)
- [Dataset Setup](https://github.com/shirley-wu/vdebugger/tree/main?tab=readme-ov-file#dataset-setup)
- [Generation and Execution of Visual Programs](https://github.com/shirley-wu/vdebugger/tree/main?tab=readme-ov-file#generation-and-execution-of-visual-programs)
- [Inference of VDebugger](https://github.com/shirley-wu/vdebugger/tree/main?tab=readme-ov-file#inference-of-vdebugger)
- [Error Injection](https://github.com/shirley-wu/vdebugger/tree/main?tab=readme-ov-file#error-injection)

## Environment Setup

This code is partially adapted from [ViperGPT](https://github.com/cvlab-columbia/viper). We sincerely thank the authors of ViperGPT for their great work!

To setup the environment, you should:
1. Clone recursively:
```bash
git clone --recurse-submodules git@github.com:shirley-wu/vdebugger.git
```
2. Install pytorch based on your own environment. We installed `torch==2.1.2` with cuda 12.1
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Setup ViperGPT environments by:
```bash
cd viper
bash download_models.sh
export PATH=/usr/local/cuda/bin:$PATH
cd GLIP
python setup.py clean --all build develop --user
```
5. If you need to use openai APIs: write api key into `viper/api.key`

## Dataset Setup

Please follow the guidelines below to download each dataset:
1. GQA: https://cs.stanford.edu/people/dorarad/gqa/download.html. The file structure should look as follows:
```
gqa/
├── questions/
│   ├── readme.txt
│   ├── {val, test, testdev, challenge}_{all, balanced}_questions.json
│   ├── submission_all_questions.json
│   ├── train_balanced_questions.json
│   ├── train_all_questions/
└── images/
    └── *.jpg
```
2. TallyQA: https://github.com/manoja328/TallyQA_dataset. The file structure should look as follows:
```
tallyqa/
├── {test, train}.json
└── {train2014, val2014, VG_100K, VG_100K_2}/
    └── *.jpg
```
3. NLVRv2: https://github.com/lil-lab/nlvr/tree/master/nlvr2. The file structure should look as follows:
```
nlvr2/
├── balanced_{dev, test1, test2, train}.jsonl
└── {dev, test1, test2, train}/
    └── *.png
```
4. RefCOCO*: https://github.com/lichengunc/refer. The file structure should look as follows:
```
refer/
├── refcoco/
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+/
│   ├── instances.json
│   └── refs(unc).p
├── refcocog/
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(umd).p
└── {train2014, train2017, val2014, val2017}/
    └── *.jpg
```
5. COVR: https://covr-dataset.github.io/. The file structure should look as follows:
```
covr/
├── {train, val, test}.jsonl
├── gqa_images/
│   └── *.jpg
└── imSitu_images/
    └── {adjusting, ...}/
        └── *.jpg
```
6. RSVG: https://github.com/ZhanYang-nwpu/RSVG-pytorch. The file structure should look as follows:
```
rsvg/
├── {train, val, test.txt}
├── Annotations/
│   └── *.xml
└── JPEGImages/
    └── *.jpg
```

## Generation and Execution of Visual Programs

Go to `viper/` for this step. We recommend first generating and then executing the visual programs in two separate steps. Take GQA dataset as an example:
1. Generate programs:
```bash
CONFIG_NAMES=generate/gqa python main_batch_generate.py
```
This script will load the configuration under `config/generate/gqa.yaml`. Please remember to change YOUR_DATA_DIR into your data directory. The generated code will be saved in a csv under `code` field
2. Execute and evaluate programs:
```bash
CONFIG_NAMES=execute/gqa python main_batch_execute.py
```
This script will load the configuration under `config/execute/gqa.yaml`. Please also remember to update YOUR_DATA_DIR, and change the `cached_codex_path:` field into the csv produced in step 1. The accuracy / IoU will be computed.
3. If you want to obtain execution feedback:
```bash
CONFIG_NAMES=execute/gqa python main_batch_trace.py A_RANDOM_STAMP
```
You can use the same configuration as in step 2. If you want to run multiple `main_batch_trace.py` in the same time, please use different `A_RANDOM_STAMP` for different processes. The execution feedback will be saved in a csv under `traced` field.

## Inference of VDebugger

For inference with VDebugger, it is required to first generate and execute visual programs, and obtain a csv file containing `traced` field. Then, go to `vdebugger/`. Take GQA dataset and VDebugger/VDebugger-{critic, refiner}-generalist-13B as an example:
```bash
# Step 1: infer critic
python infer_critic.py VDebugger/VDebugger-critic-generalist-13B --input YOUR_CSV_CONTAINING_TRACED_FIELD --dataset gqa  # output file will be written to critic-infer.csv
# Step 2: infer refiner
python infer_refine.py critic-infer.csv VDebugger/VDebugger-refiner-generalist-13B  # output file will be written to critic-refine-infer.csv
```
Then you can execute the programs in `critic-refine-infer.csv` as in step 2 of [Generation and Execution of Visual Programs](https://github.com/shirley-wu/vdebugger/tree/main?tab=readme-ov-file#generation-and-execution-of-visual-programs)

## Training of VDebugger

If you want to reproduce our training of VDebugger, please use `vdebugger/training_scripts/train_{critic, refiner}.sh`. You will need to install `deepspeed==0.14.0`.

## Error Injection

To perform error injection and generate incorrect programs as described in Section 4 of our paper, you first need a `.csv` file containing the visual programs generated for the training set and their execution results. Then, please go to `vdebugger/` and run:
```bash
python error_injection.py YOUR_CSV_FILE --error_injection {greedy, mask-best}
```

## Citation

Please cite our paper if this repository inspires your work.
```
@misc{wu2024vdebugger,
      title={VDebugger: Harnessing Execution Feedback for Debugging Visual Programs}, 
      author={Xueqing Wu and Zongyu Lin and Songyan Zhao and Te-Lin Wu and Pan Lu and Nanyun Peng and Kai-Wei Chang},
      year={2024}
}
```
