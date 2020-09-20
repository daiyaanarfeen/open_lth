import sys
import os
import pandas as pd

paths = os.listdir(sys.argv[1])
paths = [os.path.join(sys.argv[1], p, sys.argv[2], 'logger') for p in paths]
paths = paths[int(sys.argv[3]): int(sys.argv[4])+1]
dfs = [pd.read_csv(p, header=None) for p in paths]
print([max(dfs[i].loc[dfs[i][0] == "test_accuracy"][2]) for i in range(len(dfs))])
