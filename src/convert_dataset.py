import argparse
import cPickle as pickle
import numpy as np

parser = argparse.ArgumentParser(description='Convert train and test data')
parser.add_argument('rating_file',         type=str,                 help='information file')
parser.add_argument('item_list_file',      type=str,                 help='item list file path')
parser.add_argument('out_file',            type=str,                 help='output pkl file')
parser.add_argument('--min_rating',  '-m', type=float, default=None, help='minimum rating value')
parser.add_argument('--rating_unit', '-u', type=float, default=1,    help='rating unit')
args = parser.parse_args()

min_rating = args.min_rating
rating_unit = args.rating_unit
if min_rating == None:
    min_rating = rating_unit

def load_item_list(file_path):
    with open(file_path) as f:
        item_ids = [int(line.split('::')[0]) for line in f]
    item_id_to_index = np.zeros((max(item_ids) + 1,), dtype=np.int32)
    for i, item_id in enumerate(item_ids):
        item_id_to_index[item_id] = i
    return item_id_to_index

def load_data(file_path, item_id_to_index):
    with open(file_path) as f:
        lines = f.readlines()
    data_size = len(lines)
    users = np.zeros((data_size,), dtype=np.int32)
    items = np.zeros((data_size,), dtype=np.int32)
    ratings = np.zeros((data_size,), dtype=np.int32)
    timestamps = np.zeros((data_size,), dtype=np.int32)
    if item_id_to_index is None:
        item_id_to_index = np.arange()
    for i, line in enumerate(lines):
        user, item, rating, timestamp = line.split('::')
        users[i] = int(user) - 1
        items[i] = item_id_to_index[int(item)]
        rating = float(rating)
        ratings[i] = int((rating - min_rating) / rating_unit)
        timestamps[i] = int(timestamp)
    user_num = np.max(users) + 1
    item_num = np.max(items) + 1
    return user_num, item_num, (users, items, ratings, timestamps)

print 'Loading data set...'
item_id_to_index = load_item_list(args.item_list_file)
user_num, item_num, train_data = load_data(args.rating_file, item_id_to_index)

print 'Saving data set'
with open(args.out_file, 'wb') as f:
    m = {
        'user_num': user_num,
        'item_num': item_num,
        'train_data': train_data,
        'test_data': None,
        'min_rating': min_rating,
        'rating_unit': rating_unit
    }
    pickle.dump(m, f, pickle.HIGHEST_PROTOCOL)
print 'Completed'
