import numpy


def mean_average_precision(logits, target):
    """
    Compute mean average precision.
    :param logits: 2d array [batch_size x num_clicks_per_query]
    :param target: 2d array [batch_size x num_clicks_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    map = 0
    for i in range(indices.shape[0]):
        average_precision = 0
        num_rel = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                num_rel += 1
                average_precision += num_rel / (j + 1)
        average_precision = average_precision / num_rel
        map += average_precision

    return map / indices.shape[0]


def NDCG(logits, target, k):
    """
    Compute normalized discounted cumulative gain.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    assert logits.shape[1] >= k, 'NDCG@K cannot be computed, invalid value of K.'

    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    NDCG = 0
    for i in range(indices.shape[0]):
        DCG_ref = 0
        num_rel_docs = numpy.count_nonzero(target[i])
        for j in range(indices.shape[1]):
            if j == k:
                break
            if target[i, indices[i, j]] == 1:
                DCG_ref += 1 / numpy.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += 1 / numpy.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return NDCG / indices.shape[0]


def MRR(logits, target):
    """
    Compute mean reciprocal rank.
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :param target: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape

    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                reciprocal_rank += 1.0 / (j + 1)
                break

    return reciprocal_rank / indices.shape[0]
