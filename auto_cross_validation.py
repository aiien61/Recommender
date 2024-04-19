from surprise import Dataset, SVD
from surprise.model_selection import cross_validate

data = Dataset.load_builtin('ml-100k')

algo = SVD()

cross_validate(algo=algo, data=data, measures=['RMSE', 'MAE'], cv=5, verbose=True)