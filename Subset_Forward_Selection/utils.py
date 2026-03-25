from sklearn.base import clone

def clone_estimator(estimator):
     return clone(estimator)


# def select_scoring_method():

#     try:

#         methon = int(input(" Select The method That you want as numeric : \n 1-Accuracy \n 2-Recall \n 3-Precision \n 4-R2"))

#     except TypeError:

#         print("PLZ , Enter only numbers between [1 to 4]")
