
import os
import glob
import csv
import random

CATEGORY_NAMES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

def find_input_pairs(data_dir, label_data=True):
  filenames = []

  #for (dirpath, dirnames, files) in os.walk(data_dir):
  #  filenames.extend([os.path.join(dirpath, f) for f in files if ('.jpg' in f or '.jpeg' in f)])

  filenames.extend([f for f in glob.iglob(os.path.join(data_dir, '*.jpg'))])
  filenames.extend([f for f in glob.iglob(os.path.join(data_dir, '**', '*.jpg'))])
  filenames.extend([f for f in glob.iglob(os.path.join(data_dir, '*.jpeg'))])
  filenames.extend([f for f in glob.iglob(os.path.join(data_dir, '**', '*.jpeg'))])

  io_pairs = [] 
  for f in filenames:
    if(label_data):
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
    else:
      pair = { 'filename': f, 'label': 0 }
      io_pairs.append(pair)

  return io_pairs


def split_set(io_pairs, percent_test=20.0, percent_cv=20.0): 
  cv_size = int(percent_cv * len(io_pairs) / 100)
  test_size = int(percent_test * len(io_pairs) / 100)

  io_pairs = io_pairs[:]
  random.shuffle(io_pairs)
  cv = io_pairs[-cv_size:] 
  del io_pairs[-cv_size:]

  test = io_pairs[-test_size:] 
  del io_pairs[-test_size:]

  train = io_pairs

  return train, test, cv


def get_sets(data_dir, percent_test=20.0, percent_cv=20.0, label_data=True):
  io_pairs = find_input_pairs(data_dir, label_data)

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
  percent_test = 20.0
  percent_cv = 20.0

  data = find_input_pairs('train')
  (train, test, cv) = split_set(data, percent_test, percent_cv)

  submission_input = find_input_pairs('test_stg1', False)
  write_set(data, 'all.csv')
  write_set(train, 'train.csv')
  write_set(test, 'test.csv')
  write_set(cv, 'cv.csv')
  write_set(submission_input, 'submission_input.csv')

if __name__ == '__main__':
  main()
