from load_data import data
from recommender import model
from collections import namedtuple

train = data.build_full_trainset()
model.fit(trainset=train)

Target = namedtuple('Target', ['user_id', 'item_id'])
target = Target('E', 2)

prediction = model.predict(*target)
print(prediction.est)