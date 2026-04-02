import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class KNNTravelRecommender:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self._prepare_data()

    def _clean_budget(self, cost_str):
        try:
            nums = re.findall(r'\d+', str(cost_str))
            return int(nums[-1]) if nums else 0
        except:
            return 0

    def _prepare_data(self):
        self.df['max_budget'] = self.df['Trip Cost'].apply(self._clean_budget)
        
        # Super-Content String for ML Pattern Matching
        self.df['content'] = (
            self.df['Type'] + " " + 
            self.df['Place Name'] + " " + 
            self.df['Best Visit Time'] + " " + 
            self.df['Ideal For']
        ).str.lower()

    def _expand_keywords(self, user_type):
        """ Nearest Word Logic to prevent 'Not Found' """
        synonyms = {
            'trekking': 'adventure mountain hiking climbing',
            'beach': 'sea water coastal ocean sand',
            'nature': 'greenery forest hills scenic waterfall lake',
            'spiritual': 'temple religious divine holy shrine',
            'history': 'fort monument ancient heritage palace museum',
            'wildlife': 'animal safari tiger bird sanctuary national park'
        }
        
        expanded = str(user_type).lower()
        for key, val in synonyms.items():
            if key in expanded:
                expanded += " " + val
        return expanded

    def filter_hard_constraints(self, state, budget):
        """ Filter by State and Budget (Hard Rules) """
        return self.df[
            (self.df['State'].str.lower() == state.lower()) & 
            (self.df['max_budget'] <= budget)
        ].copy()

    def recommend(self, state, budget, month, travel_type, top_n=3):
        # 1. Apply Hard Filters
        df_filtered = self.filter_hard_constraints(state, budget)
        if df_filtered.empty:
            return pd.DataFrame()

        df_filtered = df_filtered.reset_index(drop=True)
        
        # 2. Expand Query (e.g. Trekking -> Adventure)
        expanded_query = self._expand_keywords(travel_type) + " " + month.lower()

        # 3. Vectorization (Text to Numbers)
        vec = TfidfVectorizer(stop_words='english')
        matrix = vec.fit_transform(df_filtered['content'])
        query_vec = vec.transform([expanded_query])
        
        # 4. KNN (K-Nearest Neighbors) Logic
        # Metric is 'cosine' distance. 
        knn = NearestNeighbors(n_neighbors=min(len(df_filtered), 10), metric='cosine')
        knn.fit(matrix)
        distances, indices = knn.kneighbors(query_vec)
        
        # 5. Convert Distances to Match Scores (Closer distance = Higher score)
        scores = np.zeros(len(df_filtered))
        for i, idx in enumerate(indices[0]):
            scores[idx] = 1 - distances[0][i]

        df_filtered['match_score'] = scores
        
        # 6. Sorting (No Randomness)
        df_filtered = df_filtered.sort_values(by=['match_score', 'Place Name'], ascending=[False, True])
        
        return df_filtered[df_filtered['match_score'] > 0].head(top_n)[['Place Name', 'City', 'Type', 'match_score']]


# ==========================================
# EVALUATION SCRIPT: Check KNN Accuracy
# ==========================================
def evaluate_knn(recommender):
    test_cases = [
        {'state': 'RJ', 'budget': 5000, 'month': 'Oct', 'type': 'trekking', 'expected': ['adventure', 'nature', 'mountain', 'hill']},
        {'state': 'CG', 'budget': 10000, 'month': 'Jul', 'type': 'waterfall', 'expected': ['waterfall', 'nature']},
        {'state': 'RJ', 'budget': 3000, 'month': 'Dec', 'type': 'history', 'expected': ['history', 'fort', 'culture', 'heritage']},
    ]
    
    print("=========================================")
    print("EVALUATING KNN RECOMMENDATION ACCURACY")
    print("=========================================\n")
    
    total_precision = 0
    K = 3
    
    for i, test in enumerate(test_cases):
        res = recommender.recommend(test['state'], test['budget'], test['month'], test['type'], top_n=K)
        
        relevant_hits = 0
        actual_results_count = len(res)
        
        if not res.empty:
            for _, row in res.iterrows():
                item_type = str(row['Type']).lower()
                if any(kw in item_type for kw in test['expected']):
                    relevant_hits += 1
                    
        precision = relevant_hits / actual_results_count if actual_results_count > 0 else 0
        total_precision += precision
        
        print(f"Test {i+1} [Query: '{test['type']}' in {test['state']}]: Accuracy = {precision*100:.2f}%")
        if not res.empty:
            print("Top Results:")
            print(res.to_string(index=False))
        else:
            print("No matching places found.")
        print("-" * 40)
        
    avg_accuracy = (total_precision / len(test_cases)) * 100
    print(f"\nOVERALL KNN MODEL ACCURACY: {avg_accuracy:.2f}%")

# Execution
if __name__ == "__main__":
    recommender = KNNTravelRecommender("final_travel_data_1000.csv")
    evaluate_knn(recommender)