from scipy import spatial
from itertools import combinations

users_ratings = {
    'user_a': [1, 2],
    'user_b': [2, 4],
    'user_c': [2.5, 4],
    'user_d': [4.5, 5]
}

pairs = combinations(users_ratings, 2)

for pair in pairs:
    ratings = (users_ratings[pair[0]], users_ratings[pair[1]])
    similarity = spatial.distance.euclidean(*ratings)
    print(f"Similarity score of {pair}: {similarity}")
