
import os
import numpy as np

def get_dirnames(data_dir):
  names = []
  for (dirpath, dirnames, files) in os.walk(data_dir):
    names.extend(dirnames)

  return np.unique(names)

def get_next_run_num(data_dir):
  dirs = get_dirnames(data_dir)

  nums = []
  for d in dirs:
    threw = False
    n = 0
    try:
      n = int(d)
    except ValueError:
      threw = True
      print('Failed to cast ' + d)

    if(~threw):
      nums.append(n)

  if len(nums) == 0:
    return 1
  else:
    return max(nums) + 1
