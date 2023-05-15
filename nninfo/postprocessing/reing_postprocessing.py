import pandas as pd

def get_directed_reing_complexity(reing_differences: pd.DataFrame):
    n = sum(reing_differences.columns.str.match(r'C\(\d+\|\|\d+\)'))

    complexity = pd.DataFrame()
    complexity['run_id'] = reing_differences['run_id']
    complexity['chapter_id'] = reing_differences['chapter_id']
    complexity['epoch_id'] = reing_differences['epoch_id']
    complexity['reing_complexity'] = sum(k * reing_differences[f'C({k}||{k+1})'] for k in range(1, n)) / sum(reing_differences[f'C({k}||{k+1})'] for k in range(1, n))
    return complexity