import argparse
import cPickle as pickle
import numpy as np

parser = argparse.ArgumentParser(description='Convert train and test data')
parser.add_argument('rating_file',         type=str,                 help='information file')
parser.add_argument('out_file',            type=str,                 help='output pkl file')
parser.add_argument('--min_rating',  '-m', type=float, default=None, help='minimum rating value')
parser.add_argument('--rating_unit', '-u', type=float, default=1,    help='rating unit')
args = parser.parse_args()

min_rating = args.min_rating
rating_unit = args.rating_unit
if min_rating == None:
    min_rating = rating_unit

def load_data(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    data_size = len(lines)
    users = np.zeros((data_size,), dtype=np.int32)
    items = np.zeros((data_size,), dtype=np.int32)
    ratings = np.zeros((data_size,), dtype=np.float32)
    timestamps = np.zeros((data_size,), dtype=np.int32)
    for i, line in enumerate(lines):
        user, item, rating, timestamp = line.split('::')
        users[i] = int(user) - 1
        items[i] = int(item) - 1
        rating = float(rating)
        ratings[i] = int((rating - min_rating) / rating_unit)
        timestamps[i] = int(timestamp)
    user_num = np.max(users) + 1
    item_num = np.max(items) + 1
    return user_num, item_num, (users, items, ratings, timestamps)

print 'Loading data set...'
user_num, item_num, train_data = load_data(args.rating_file)

print 'Saving data set'
with open(args.out_file, 'wb') as f:
    pickle.dump((user_num, item_num, train_data, None), f, pickle.HIGHEST_PROTOCOL)
print 'Completed'
