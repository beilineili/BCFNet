import scipy.sparse as sp
import numpy as np
import csv


class Dataset(object):
    '''
    classdocs
    '''
    def __init__(self, path):
        '''
        Constructor
        '''
        # 训练集-矩阵
        self.trainMatrix = self.load_rating_file_as_matrix(path +
                                                           ".train_rating.csv")
        # 测试评分-线性表
        self.testRatings = self.load_rating_file_as_list(path +
                                                         ".test_rating.csv")
        # 测试用负样例-线性表
        self.testNegatives = self.load_negative_file(path +
                                                     ".test_negative.csv")
        assert len(self.testRatings) == len(self.testNegatives)

        self.num_users, self.num_items = self.trainMatrix.shape

    # 将用户和item 拉平成一个线性表
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split(",")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    # 要先看看数据集是什么样的格式
    # 数据集文件本身就是随机取样好了的，这里只是把它拉平成一个线性表
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split(",")
                negatives = []
                # 除了第一个括号之后的全部都是没评分过的随机取样的item
                for x in arr[2:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split(",")
                u, i = int(arr[0]), int(arr[1])
                # 看用户编号，有更大的编号就意味着有更大的用户数量
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        # 建矩阵，如果用户有item的评分就设置那个位置为1，之所以要加一是因为用户编号第一个为0
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            # while line != None and line != "":
            while line is not None and line != "":
                arr = line.split(",")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    mat[user, item] = 1.0
                line = f.readline()
        return mat
