from surprise import Dataset, SVD
from surprise.model_selection import cross_validate
from surprise import Reader

import pandas as pd
from pprint import pprint
from toolz import pipe

# reader = Reader(line_format='user item rating timestamp',
#                 sep=',', skip_lines=1)
# data = Dataset.load_from_file('./dataset/ml-latest-small/ratings.csv', reader=reader)
# print(data)

# model = SVD()
# cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


with open('./dataset/ml-100k/u.data') as file:
    ratings = pipe(map(lambda line: line.split("\t")[:3], file),
                   lambda data: pd.DataFrame(data))
    ratings.columns = ['user_id', 'movie_id', 'rating']

print(ratings.head())
print(ratings.shape)
print(len(ratings['user_id'].unique()))
print(len(ratings.groupby(['movie_id'])))


