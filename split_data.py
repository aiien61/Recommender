import pandas as pd
from toolz import pipe

with open('./dataset/ml-100k/u.data') as file:
    ratings = pipe(map(lambda line: line.split("\t")[:3], file),
                   lambda data: pd.DataFrame(data))
    ratings.columns = ['user_id', 'movie_id', 'rating']

print(ratings.head())
print(ratings.shape)
print(len(ratings['user_id'].unique()))
print(len(ratings.groupby(['movie_id'])))
