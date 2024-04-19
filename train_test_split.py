from surprise import SVD, Dataset, accuracy
from surprise.model_selection import train_test_split

data = Dataset.load_builtin('ml-100k')

trainingset = {}

for size in [0.1, 0.15, 0.2, 0.25, 0.3]:
    trainset, testset = train_test_split(data, test_size=size)
    trainingset[size] = trainset
    
    algo = SVD()
    predictions = algo.fit(trainset).test(testset)
    
    # Compute RMSE, MAE, MSE
    print('Testset size:', size)
    accuracy.rmse(predictions=predictions)
    accuracy.mae(predictions=predictions)
    accuracy.mse(predictions=predictions)

algo = SVD()
algo.fit(trainingset[0.2])

uid = str(196)
iid = str(302)
algo.predict(uid, iid, r_ui=4, verbose=True)