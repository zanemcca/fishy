
import os
import csv
import random

CATEGORY_NAMES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def find_input_pairs(data_dir):
  filenames = []
  for (dirpath, dirnames, files) in os.walk(data_dir):
    filenames.extend([os.path.join(dirpath, f) for f in files if ('.jpg' in f or '.jpeg' in f)])

  io_pairs = [] 
  for f in filenames:
    label =-1 
    for i, c in enumerate(CATEGORY_NAMES):
      if c in f:
        label = i
        break

    if(label >= 0):
      pair = { 'filename': f, 'label': label }
      io_pairs.append(pair)
    else:
      raise ValueError('No label found for ' + f)

  return io_pairs


def split_set(io_pairs, percent_test=20.0, percent_cv=20.0): 
  cv_size = int(percent_cv * len(io_pairs) / 100)
  test_size = int(percent_test * len(io_pairs) / 100)

  print(len(io_pairs))
  random.shuffle(io_pairs)
  print(len(io_pairs))
  print(cv_size)
  cv = io_pairs[-cv_size:] 
  del io_pairs[-cv_size:]

  test = io_pairs[-test_size:] 
  del io_pairs[-test_size:]

  train = io_pairs

  return train, test, cv


def get_sets(data_dir, percent_test=20.0, percent_cv=20.0):
  io_pairs = find_input_pairs(data_dir)

  return split_set(io_pairs, percent_test, percent_cv)


def write_set(io_pairs, filename):
  with open(filename, 'wb') as f:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(io_pairs)


def read_set(filename):
  with open(filename, 'rb') as f:
    reader = csv.DictReader(f)
    io_pairs = [{ 'filename': p['filename'], 'label': int(p['label'])} for p in reader]

  return io_pairs


def main(argv=None):
  # TODO Get percent_test and percent_cv from input arguments
  percent_test = 20.0
  percent_cv = 20.0
  (train, test, cv) = get_sets(os.path.join(os.getcwd(), 'train'), percent_test, percent_cv)

  write_set(train, 'train.csv')
  write_set(test, 'test.csv')
  write_set(cv, 'cv.csv')

if __name__ == '__main__':
  main()
