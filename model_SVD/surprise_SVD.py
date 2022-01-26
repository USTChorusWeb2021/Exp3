from operator import mod
from numpy import double
from surprise import accuracy
from surprise import BaselineOnly,  KNNBasic, NormalPredictor
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
from queue import PriorityQueue

reader = Reader(rating_scale=(1, 5), line_format='user item rating', sep=';')
data = Dataset.load_from_file('./surprise_read_data1.txt', reader=reader)
trainset, testset = train_test_split(data, test_size=0.0000001)

# algo1 = SVD()
# algo2 = SVD(biased = False)
# algo3 = SVDpp()

# algo1.fit(data)

model = SVD(n_factors=30)
model.fit(trainset)
print(model.predict(2,14))
print(model.predict(3,1))

print(model.pu.shape)
print(model.qi.shape)

USER_NUMBER = 23599
ITEM_NUMBER = 21602

def get_top_n(n=10):

    # First map the predictions to each user.
    
    model = SVD(n_factors=30)
    model.fit(trainset)
    top_n = PriorityQueue()
    # uid： 用户ID
    # iid： item ID
    # true_r： 真实得分
    # est：估计得分
    for uid in range(0, USER_NUMBER):
        print(uid, "\t")
        for iid in range(0, ITEM_NUMBER):
            print(model.predict(uid, iid))
            top_n.put([model.predict(uid, iid).est, iid])
            if (top_n.qsize() > n):
                top_n.get()
        recommand_list = []
        while not top_n.empty():
            t = top_n.get()
            recommand_list.insert(0, t[1])
        while len(recommand_list) != 1:
            print(recommand_list.pop(0),",")
        print(recommand_list.pop(0),"\n")
        
# get_top_n()