import numpy as np
from utils import clone_estimator
from metrics import get_score

class SequentialFeatureSelector :
    def __init__(self , estimator , n_features_to_select , direction ="forward" , scoring="accuracy") :

        self.estimator = estimator
        self.n_feature_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring

        self.selected_features = None


    def evaluate_subset(self , X , y , feature_indices) :
        
        model_cloned = clone_estimator(self.estimator) 
        X_subset=X[ : , feature_indices]

        model_cloned.fit(X_subset , y)
        y_pred = model_cloned.predict(X_subset)

        score = get_score( y , y_pred , self.scoring)

        return(score)
    

        
    def fit (self , X , y):
        self.n_features = X.shape[1]

        if self.direction == "forward" :
            selected = []
            remaining = list(range(self.n_features))

            while len(selected) < self.n_feature_to_select :
                best_score = -np.inf
                best_feature = None

                for f in remaining : 
                    current_features = selected + [f]
                    score = self.evaluate_subset(X , y , current_features)

                    if score > best_score :
                        best_score = score
                        best_feature = f

                selected.append(best_feature)
                remaining.remove(best_feature)

            self.selected_features = selected

        elif self.direction == "backward" :
            selected = list(range(self.n_features))

            while len(selected) > self.n_feature_to_select :
                best_score = -np.inf
                worst_feature = None

                for f in selected :
                    current_features = [feature for feature in selected if feature != f]
                    score = self.evaluate_subset(X , y , current_features)

                    if score > best_score : 
                        best_score = score
                        worst_feature = f

                selected.remove (worst_feature)

            self.selected_features = selected

        return self
    

    def transform(self , X) :
        return X[ : ,self.selected_features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    

    def get_support(self):
        mask = [False] * self.n_features
    
        for idx in self.selected_features:
            mask[idx] = True
    
        return mask
