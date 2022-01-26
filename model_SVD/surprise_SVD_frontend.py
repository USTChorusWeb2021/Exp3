from calendar import EPOCH
from tabnanny import verbose
from surprise import accuracy
from surprise import BaselineOnly,  KNNBasic, NormalPredictor
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import cross_validate, train_test_split
import numpy

USER_NUMBER = 23599
ITEM_NUMBER = 21602

reader = Reader(rating_scale=(0, 1), line_format='user item rating', sep=';')
data = Dataset.load_from_file('./surprise_read_data_2.5.txt', reader=reader)
trainset, testset = train_test_split(data, test_size=0.0000001)

# algo1 = SVD()
# algo2 = SVD(biased = False)
# algo3 = SVDpp()

# algo1.fit(data)

model = SVD(n_factors=30, verbose=True, biased=True, n_epochs=100)
model.fit(trainset)

print(model.pu.shape)
print(model.qi.shape)

pu: numpy.ndarray = model.pu
qi: numpy.ndarray = model.qi
bi: numpy.ndarray = model.bi

# rbar = pu.dot(qi.transpose())

# print(rbar.shape)

numpy.savetxt("item_bias.txt", bi)
numpy.savetxt("predict_matrix_p.txt", pu)
numpy.savetxt("predict_matrix_q.txt", qi.transpose())

