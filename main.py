
import pandas as pd
from pprint import pprint

from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from toolz import pipe

### load dataset
print('loading data'.ljust(30, '.'), end='')
path_ratings = './dataset/ml-latest-small/ratings.csv'
path_items = './dataset/ml-latest-small/movies.csv'

ratings_df = pd.read_csv(path_ratings)
items_df = pd.read_csv(path_items)

config = {'dataframe': {'ratings': ratings_df, 'items': items_df}}

# print(ratings_df.head())
# print(items_df.head())

user_col = 'userId'
item_col = 'movieId'
item_types_col = 'genres'
item_name_col = 'title'

config.update({
    'columns': {
        'user': 'userId', 
        'item': 'movieId', 
        'item_type': 'genres',
        'item_name': 'title'
    }
})
print('completed')

### preprocess
print('preprocessing'.ljust(30, '.'), end='')
df = pd.merge(ratings_df, items_df[[item_col, item_types_col]], on=item_col, how='left')
# print(df.head())
# print(df.dtypes)

# label encoder responsible for user and item columns, one-hot encoder for items' types
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
mlb = MultiLabelBinarizer()

config.update({
    'encoder': {
        'user': user_encoder,
        'item': item_encoder,
        'item_type': mlb
    }
})

df[user_col] = user_encoder.fit_transform(df[user_col])
df[item_col] = item_encoder.fit_transform(df[item_col])
# print(df.head())
# print(df.dtypes)

# print(df[item_types_col].str.split('|'))
# print(mlb.fit_transform(df[item_types_col].str.split('|')))
df = pipe(df.pop(item_types_col).str.split('|'),
          mlb.fit_transform,
          lambda x: pd.DataFrame(x, columns=mlb.classes_, index=df.index),
          df.join)

# print(df.head())
# print(df.columns)

config['dataframe'].update({'merged': df})

# drop the useless column
df.drop(columns="(no genres listed)", inplace=True)
# print(df.head)
# print(df.columns)
# print(pd.DataFrame.describe(df['rating']))

# split dataset into training and test sets
train_df, test_df = train_test_split(df, test_size=0.2)
# print(df.shape)
# print(train_df.shape)
# print(test_df.shape)

# build trainset instance using reader and Dataset and full training data
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(train_df[[user_col, item_col, 'rating']], reader=reader)
trainset = data.build_full_trainset()
# print(trainset)
print('completed')


### modelling 
algo = SVD()
print('model training'.ljust(30, '.'), end='')
algo.fit(trainset)

config.update({'model': algo})
print('completed')

### evaluation
print('model evaluation'.ljust(30, '.'), end='')
predictions = algo.test(trainset.build_anti_testset())
print('completed')
# pprint(predictions[:5])

# print(predictions[0])
# print(predictions[0].uid, predictions[0].iid,
#       predictions[0].r_ui, predictions[0].est)

def get_top_n(user_id: str, n=5, **kwargs):
    df = kwargs['dataframe']['merged']
    items_df = kwargs['dataframe']['items']
    algo = kwargs['model']
    item_encoder = kwargs['encoder']['item']
    user_col = kwargs['columns']['user']
    item_col = kwargs['columns']['item']
    item_name_col = kwargs['columns']['item_name']
    

    user_items = df[df[user_col] == user_id][item_col].unique()  # items which have been rated by the user
    all_items = df[item_col].unique()
    items_to_predict = list(set(all_items) - set(user_items)) # items which haven't been rated by the user yet

    user_item_pairs = [(user_id, item_id, 0) for item_id in items_to_predict]
    predictions_cf = algo.test(user_item_pairs)
    top_n_recommendations = sorted(predictions_cf, key=lambda x: x.est, reverse=True)[:n]

    predicted_rating = []
    for pred in top_n_recommendations:
        predicted_rating.append(pred.est)
    
    top_n_item_ids = [int(pred.iid) for pred in top_n_recommendations]
    top_n_items = item_encoder.inverse_transform(top_n_item_ids)
    top_n_items = items_df[items_df[item_col].isin(top_n_items)][item_name_col].tolist()
    print(f'Top {n} Recommendations for User {user_id}: ')
    for i, name in enumerate(top_n_items):
        predicted_item = name.ljust(50, '.')
        print(f"{i+1}.{predicted_item} estimated rating: {predicted_rating[i]}")
    
    return None

if __name__ == '__main__':
    get_top_n(user_id=1, **config)