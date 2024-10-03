import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('orig')
parser.add_argument('new')
args = parser.parse_args()

orig = pd.read_csv(args.orig)
new = pd.read_csv(args.new)
new.loc[new['code'] == orig['code'], 'code'] = '['
print("Remaining valid code", (new['code'] != '[').sum())

out_fname = args.new.replace('.csv', '.remove-dup.csv')
print("Dump to", out_fname)
assert not os.path.exists(out_fname), "File {} exists".format(out_fname)
new.to_csv(out_fname)
