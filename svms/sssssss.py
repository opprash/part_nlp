import os
import importlib,sys
import matplotlib.cm as cm
import numpy as np
import codecs
import jieba
import math
import jieba.analyse
import matplotlib.pyplot as plt


def read_from_file(file_name):
    with open(file_name, "r") as fp:
        words = fp.read()
    return words


def stop_words(stop_word_file):
    words = read_from_file(stop_word_file)
    result = jieba.cut(words)
    new_words = []
    for r in result:
        #print(r)
        new_words.append(r)
    return set(new_words)


def gen_sim(A, B):
    num = float(np.dot(A, B.T))
    denum = np.linalg.norm(A) * np.linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn
    return sim


def del_stop_words(words, stop_words_set):
    #   words是已经切词但是没有去除停用词的文档。
    #   返回的会是去除停用词后的文档
    result = jieba.cut(words)
    new_words = []
    for r in result:
        if r not in stop_words_set:
            new_words.append(r)
            # print r.encode("utf-8"),
    # print len(new_words),len(set(new_words))
    return new_words


def tfidf(term, doc, word_dict, docset):
    tf = float(doc.count(term)) / (len(doc) + 0.001)
    idf = math.log(float(len(docset)) / word_dict[term])
    return tf * idf


def idf(term, word_dict, docset):
    idf = math.log(float(len(docset)) / word_dict[term])
    return idf


def word_in_docs(word_set, docs):
    word_dict = {}
    for word in word_set:
        # print word.encode("utf-8")
        word_dict[word] = len([doc for doc in docs if word in doc])
        # print word_dict[word],
    return word_dict


def get_all_vector(path, stop_words_set):
    #names = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    #posts = [open(name).read() for name in names]
    file = codecs.open(path, 'r')
    lines = [line.strip() for line in file]
    str = ''.join(lines)
    print(len(lines))
    print('over')
    docs = []
    word_set = set()
    for each in lines:
        doc = del_stop_words(each, stop_words_set)
        #print(doc)
        #print(doc)
        docs.append(doc)
        word_set |= set(doc)
        #print(word_set)
        # print len(doc),len(word_set)

    word_set = list(word_set)
    print(word_set)
    docs_vsm = []
    # for word in word_set[:30]:
    # print word.encode("utf-8"),
    #print(docs)

    #for each in docs:
        #print(each)

    for doc in docs:
        a=0
        #print(len(doc))
        temp_vector = []
        for word in word_set:
            if word in doc:
                #print("yes+"+a.__str__())
                a+=1
            #print(doc.count(word) * 1.0)
            #print(word)
            temp_vector.append(doc.count(word) * 1.0)
            #print(doc.count(word) * 1.0)
        # print temp_vector[-30:-1]
        docs_vsm.append(temp_vector)
        #print(docs_vsm)
    docs_matrix = np.array(docs_vsm)
    # print docs_matrix.shape
    # print len(np.nonzero(docs_matrix[:,3])[0])
    column_sum = [float(len(np.nonzero(docs_matrix[:, i])[0])) for i in range(docs_matrix.shape[1])]
    column_sum = np.array(column_sum)
    column_sum = docs_matrix.shape[0] / column_sum
    #print(column_sum)
    idf = np.log(column_sum)

    idf = np.diag(idf)
    #print(idf.shape)
    #print(idf)
    # print idf.shape
    # row_sum    = [ docs_matrix[i].sum() for i in range(docs_matrix.shape[0]) ]
    # print idf
    # print column_sum
    for doc_v in docs_matrix:
        if doc_v.sum() == 0:
            doc_v = doc_v / 1
        else:
            doc_v = doc_v / (doc_v.sum())
    #print(docs_matrix)
    tfidf = np.dot(docs_matrix, idf)
    print(len(word_set))
    print(tfidf.shape)
    #print(type(tfidf))
    np.delete(tfidf,[0],axis=1)
    tfidf_mean=[]
    tfidf_max = []
    for m in range(len(word_set)):
        s=0.
        for r in range(len(lines)):
            s+=tfidf[r,m]
            #print(tfidf[r,m])
        #print(s)
        tfidf_mean.append(s)
    print(tfidf_mean)
    print("sssssss")
    dele_axis=[]
    ss=0
    for j in range(len(tfidf_mean)):
        means = np.mean(tfidf_mean)
        if tfidf_mean[j]<means:
            dele_axis.append([j])
        #tfidf_mean.remove(j)
    tfidf = np.delete(tfidf, dele_axis, axis=1)
        #print(ss)
    print(tfidf.shape)
    return  tfidf


def PCA(weight, dimension):
    from sklearn.decomposition import PCA

    print('原有维度: ', len(weight[0]))

    print('开始降维:')


    pca = PCA(n_components=dimension)  # 初始化PCA
    X = pca.fit_transform(weight)  # 返回降维后的数据
    print('降维后维度: ', len(X[0]))

    print(X)

    return X

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids
def kMeans(dataSet, k, distMeas=gen_sim, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    counter = 0
    while counter <= 50:
        counter += 1
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # print centroids
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # assign centroid to mean
    return centroids, clusterAssment


def Silhouette(X, y):
    from sklearn.metrics import silhouette_samples, silhouette_score

    print
    '计算轮廓系数:'

    silhouette_avg = silhouette_score(X, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X, y)  # 每个点的轮廓系数

    #pprint(silhouette_avg)

    return silhouette_avg, sample_silhouette_values


def Draw(silhouette_avg, sample_silhouette_values, y, k,X):


    # 创建一个 subplot with 1-row 2-column
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)

    # 第一个 subplot 放轮廓系数点
    # 范围是[-1, 1]
    ax1.set_xlim([-0.2, 0.5])

    # 后面的 (k + 1) * 10 是为了能更明确的展现这些点
    ax1.set_ylim([0, len(X) + (k + 1) * 10])

    y_lower = 10

    for i in range(k):  # 分别遍历这几个聚类

        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / k)  # 搞一款颜色
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0,
                          ith_cluster_silhouette_values,
                          facecolor=color,
                          edgecolor=color,
                          alpha=0.7)  # 这个系数不知道干什么的

        # 在轮廓系数点这里加上聚类的类别号
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # 计算下一个点的 y_lower y轴位置
        y_lower = y_upper + 10

        # 在图里搞一条垂直的评论轮廓系数虚线
    ax1.axvline(x=silhouette_avg, color='red', linestyle="--")

    plt.show()

"""
def draw(datMat,clusterAssing,myCentroids):
    marker = ['s', 'o', '^', '<']  # 散点图点的形状
    color = ['b', 'm', 'c', 'g']  # 颜色
    X = np.array(datMat)  # 数据点
    #print(X)
    CentX = np.array(myCentroids)  # 质心点4个
    Cents = np.array(clusterAssing[:, 0])  # 每个数据点对应的簇
    for i, Centroid in enumerate(Cents):  # 遍历每个数据对应的簇，返回数据的索引即其对应的簇
            plt.scatter(X[i][0], X[i][1], marker=marker[int(Centroid[0])], c=color[int(Centroid[0])])  # 按簇画数据点
    plt.scatter(CentX[:, 0], CentX[:, 1], marker='*', c='r')  # 画4个质心
    plt.show()
"""

if __name__ == "__main__":
    path = 'C:\\Users\\ruanlu\\Desktop\\test.txt'
    path1='C:\\Users\\ruanlu\\Desktop\\stop_words.txt'
    """
    stop_words = stop_words("./stop_words.txt")
    names, tfidf_mat = get_all_vector("./chinese/", stop_words)
    myCentroids, clustAssing = kMeans(tfidf_mat, 3, gen_sim, randCent)
    for label, name in zip(clustAssing[:, 0], names):
        print (label, name)
    """
    stop_words = stop_words(path1)
    tfidf_mat = get_all_vector(path, stop_words)
    #
    myCentroids, clustAssing = kMeans(tfidf_mat, 4, gen_sim, randCent)
    #print(len(clustAssing))
    #for label in clustAssing[:,0]:
       # print(label)
    tfidf_mat = PCA(tfidf_mat, 2)
    draw(tfidf_mat,clustAssing,myCentroids)
    #print(tfidf())
    #print(stop_words())