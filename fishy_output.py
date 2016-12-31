
import os
import numpy as np
import csv
import fishy_constants as const

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

def read_output(filename):
  data = []
  with open(filename, 'rb') as f:
    reader = csv.DictReader(f)
    for row in reader:
      data.append(row)

  return data

def write_output(filename, data):
  if(len(data) > 0):
    with open(filename, 'wb') as f:
      fieldnames = ['image']
      for c in const.CLASSES:
        fieldnames.append(c)

      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for row in data:
        writer.writerow(row)
