"""
Hybrid Recommendation module.

Person B:
Combines Content-Based and Collaborative Filtering.
"""

class HybridRecommender:

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def combine(self, cb_scores, cf_scores):
        """
        Combine Content-Based and Collaborative scores.
        """
        pass

    def recommend(self, user_id, top_k=10):
        """
        Generate Top-K hybrid recommendations.
        """
        pass
