import os
from surprise import Reader
from surprise import Dataset
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV, PredefinedKFold
from surprise import accuracy, SVD

data = Dataset.load_builtin('ml-100k')

sim_options = {
    'name': ['msd', 'cosine'],
    'min_support': [3, 4, 5],
    'user_based': [True, False]
}
# param_grid = {'sim_options': sim_options}
param_grid = {"n_epochs": [5, 10], "lr_all": [0.002, 0.005], "reg_all": [0.4, 0.6]}
gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
    

# reader = Reader('ml-100k')
# file_directory = './dataset/ml-100k'

# # folds_files is a list of tuples containing file paths:
# # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
# train_file = os.path.join(file_directory, 'u{i}.base')
# test_file = os.path.join(file_directory, 'u{i}.test')

# folds_files = [(train_file.format(i=i), test_file.format(i=i))
#                for i in range(1, 6)]
# data = Dataset.load_from_folds(folds_files, reader=reader)
# pkf = PredefinedKFold()

# algo = SVD()

# for trainset, testset in pkf.split(data):

#     # train and test algorithm.
#     algo.fit(trainset)
#     predictions = algo.test(testset)

#     # Compute and print Root Mean Squared Error
#     accuracy.rmse(predictions, verbose=True)
