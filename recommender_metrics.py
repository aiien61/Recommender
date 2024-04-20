from itertools import combinations
from collections import defaultdict

from surprise import accuracy

class RecommenderMetrics:
    def mae(predictions):
        return accuracy.mae(predictions, verbose=True)
    
    def rmse(predictions):
        return accuracy.rmse(predictions, verbose=True)
    
    def get_top_n(predictions, n=10, minimum_rating=4.0) -> dict:
        top_n = defaultdict(list)

        for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
            if estimated_rating >= minimum_rating:
                top_n[int(user_id)].append((int(movie_id), estimated_rating))

        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n
    
    def hit_rate(top_n_predicted, leftout_predictions):
        hits, total = 0, 0

        # for each left-out rating
        for leftout in leftout_predictions:
            user_id = leftout[0]
            leftout_movie_id = leftout[1]
            # is it in the predicted top 10 for this user?
            hit = False
            for movie_id, prediction_rating in top_n_predicted[int(user_id)]:
                if int(leftout_movie_id) == int(movie_id):
                    hit = True
                    break
            
            if hit:
                hits += 1
            
            total += 1

        # compare overral precision
        return hits / total
    
    def cumulative_hit_rate(top_n_predicted, leftout_predictions, rating_cutoff=0):
        hits, total = 0, 0
        
        # for each left-out rating
        for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_predictions:
            # only look at ability to recommend things the users actually liked
            if actual_rating >= rating_cutoff:
                hit = False
                for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                    if int(leftout_movie_id) == movie_id:
                        hit = True
                        break
                if hit:
                    hits += 1
                
                total += 1

        # compute overral precision
        return hits / total
    
    def rating_hit_rate(top_n_predicted, leftout_predictions):
        hits = defaultdict(float)
        total = defaultdict(float)

        # for each left-out rating
        for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_predictions:
            # is it in the predicted top n for this user?
            for movie_id, _ in top_n_predicted[int(user_id)]:
                if int(leftout_movie_id) == movie_id:
                    hit = True
                    break
            
            if hit:
                hits[actual_rating] += 1
        
        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])

    def average_reciprocal_hit_rank(top_n_predicted, leftout_predictions):
        summation, total = 0, 0
        
        # for each left-out rating
        for user_id, leftout_movie_id, actual_rating, estimated_rating, _ in leftout_predictions:
            # is it in the predicted top n for this user?
            hit_rank = 0
            rank = 0
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                rank += 1
                if int(leftout_movie_id) == movie_id:
                    hit_rank = rank
                    break
            
            if hit_rank > 0:
                summation += (1.0 / hit_rank)

        return summation / total
    

    # what percentage of users have at least one 'good' recommendation
    def user_coverage(top_n_predicted, num_users, rating_threshold=0):
        hits = 0
        for user_id in top_n_predicted.keys():
            hit = False
            for movie_id, predicted_rating in top_n_predicted[user_id]:
                if predicted_rating >= rating_threshold:
                    hit = True
                    break
            
            if hit:
                hits += 1
        
        return hits / num_users
    

    def diversity(top_n_predicted, similarity_algo):
        n = 0
        total = 0
        similarity_matrix = similarity_algo.compute_similarities()
        for user_id in top_n_predicted.keys():
            pairs = combinations(top_n_predicted[user_id], 2)
            for pair in pairs:
                movie1 = pair[0][0]
                movie2 = pair[1][0]
                inner_id1 = similarity_algo.trainset.to_inner_iid(str(movie1))
                inner_id2 = similarity_algo.trainset.to_inner_iid(str(movie2))
                similarity = similarity_matrix[inner_id1][inner_id2]
                total += similarity
                n += 1

        S = total / n
        return (1 - S)
    
    def novelty(top_n_predicted, rankings):
        n = 0
        total = 0
        for user_id in top_n_predicted.keys():
            for rating in top_n_predicted[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1

        return total / n
