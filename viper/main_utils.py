import pathlib


def my_collate(batch):
    # Avoid stacking images (different size). Return everything as a list
    to_return = {k: [d[k] for d in batch] for k in batch[0].keys()}
    return to_return


class CompileTimeError(Exception):
    pass


class ProgramRuntimeError(Exception):
    pass


class TimeOutException(Exception):
    pass


def save_results_csv(config, df, filename=None):
    results_dir = pathlib.Path(config['results_dir'])
    results_dir = results_dir / config.dataset.split
    results_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        if not config.save_new_results:
            filename = 'results.csv'
        else:
            existing_files = list(results_dir.glob('results_*.csv'))
            if len(existing_files) == 0:
                filename = 'results_0.csv'
            else:
                filename = 'results_' + str(max([int(ef.stem.split('_')[-1]) for ef in existing_files if
                                                 str.isnumeric(ef.stem.split('_')[-1])]) + 1) + '.csv'
        filename = results_dir / filename

    print('Saving results to', filename)
    df.to_csv(filename, header=True, index=False, encoding='utf-8')

    return str(filename)
