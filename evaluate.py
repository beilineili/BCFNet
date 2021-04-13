import math
import heapq  # for retrieval topK
import multiprocessing
import numpy as np

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None


def evaluate_model(model, testRatings, testNegatives, K, num_thread):
    """
    Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    #topK 预测的K 也就是HR@K那里的参数
    _K = K

    hits, ndcgs = [], []
    if num_thread > 1:  # Multi-thread
        # 建了个池用来处理多线程，将第二个的元素一个个作为参数map进第一个函数参数中
        pool = multiprocessing.Pool(processes=num_thread)
        res = pool.map(eval_one_rating, range(len(_testRatings)))
        pool.close()
        # 等上一个线程结束才会开启下一个
        pool.join()
        hits = [r[0] for r in res]
        ndcgs = [r[1] for r in res]
        return (hits, ndcgs)
    # Single thread
    for idx in range(len(_testRatings)):
        (hr, ndcg) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)
    # 最后的评分是一个线性表
    return (hits, ndcgs)


def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItem = rating[1]
    # 在负取样的基础上加进了该用户评分的item
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    # 为了可以让下面的预测数量一致，都是数组
    users = np.full(len(items), u, dtype='int32')
    # 把这个用户全是当前用户，item是负取样的item加上当前用户的item的测试集丢给model去得到N个预测值
    # 即这些item用户喜欢的概率分别是多少
    predictions = _model.predict([users, np.array(items)],
                                 batch_size=100,
                                 verbose=0)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()

    # Evaluate top rank list
    # 得到用户喜欢item概率最高的K个
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)


# 当前用户评过分的item在topK里就是1，否则就是0
def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0
