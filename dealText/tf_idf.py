# -*- coding: utf-8 -*-
import os
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
#from sklearn.cluster import isodata
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import codecs
import jieba
import math
import jieba.analyse



def read_from_file(file_name):
    with open(file_name,mode='r') as fp:
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

def tfidf(path, stop_words_set):
    file = codecs.open(path, 'r','utf-8')
    lines = [line.strip() for line in file]
    str = ''.join(lines)
    print(len(lines))
    print('over')
    docs = []
    word_set = set()
    for each in lines:
        doc = del_stop_words(each, stop_words_set)
        docs.append(doc)
       # word_set |= set(doc)
    #word_set = list(word_set)
    words=[]
    for each in docs:
        str=''
        for k in each:
            str+=k+' '
        words.append(str)

    print(word_set)
    print(docs)
    print(len(docs))

    # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = CountVectorizer(max_features=100)
    # 该类会统计每个词语的tf-idf权值
    tf_idf_transformer = TfidfTransformer()
    # 将文本转为词频矩阵并计算tf-idf
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(words))
    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    x_train_weight = tf_idf.toarray()

    print(x_train_weight)
    return x_train_weight,lines


def get_all_vector(path, stop_words_set):
    file = codecs.open(path, 'r')
    lines = [line.strip() for line in file]
    str = ''.join(lines)
    print(len(lines))
    print('over')
    docs = []
    word_set = set()
    for each in lines:
        doc = del_stop_words(each, stop_words_set)
        docs.append(doc)
        word_set |= set(doc)
    word_set = list(word_set)
    print(word_set)
    docs_vsm = []
    for doc in docs:
        a=0
        #print(len(doc))
        temp_vector = []
        for word in word_set:
            if word in doc:
                a+=1
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
    l=len(idf)
    m=np.zeros((l,l))
    for each in l:
        m[l,l]=idf[l]
    print(idf)
    #idf = np.diag(idf)
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
    tfidf = np.dot(docs_matrix, m)
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
    # 数据小的时候
    # for j in range(len(tfidf_mean)):
    #     means = np.mean(tfidf_mean)
    #     if tfidf_mean[j]<means:
    #         dele_axis.append([j])
        #tfidf_mean.remove

    #新增top k
    dele_axis_top=[]
    tfidf1 = sorted(tfidf_mean)
    tfidf_max=tfidf1[len(tfidf_mean)-1000-1:]
    for j in range(len(tfidf_mean)):
        if tfidf_mean[j] not in tfidf_max:
            dele_axis_top.append([j])

    # 新增top k
    tfidf=np.delete(tfidf,dele_axis_top,axis=1)

    #数据小的时候
    #tfidf = np.delete(tfidf, dele_axis, axis=1)
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


def kmeans(X, k):  # X=weight
    clusterer = KMeans(n_clusters=k, init='k-means++',max_iter=1000)  # 设置聚类模型
    y = clusterer.fit_predict(X)  # 把weight矩阵扔进去fit一下,输出label
    print('ss')
    print(type(y))
    print(y.shape)
    return y
def Silhouette(X, y):
    print
    '计算轮廓系数:'

    silhouette_avg = silhouette_score(X, y)  # 平均轮廓系数
    sample_silhouette_values = silhouette_samples(X, y)  # 每个点的轮廓系数

    #pprint(silhouette_avg)

    return silhouette_avg, sample_silhouette_values


def birch(X, k):  # 待聚类点阵,聚类个数

    from sklearn.cluster import Birch
    clusterer = Birch(n_clusters=k)
    y = clusterer.fit_predict(X)


    return y
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

        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i)/ k)
       # color = cm.spectral(float(i) / k)  # 搞一款颜色
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
def mkdir(path):
    folder=os.path.exists(path)
    if not folder:
        os.mkdir(path)
        print(' -------new folder-----')
        print("------ok------")
    else:
        print("here is already a folder")

def sav_ressult(y,path2,k,lines):
    dics={}
    line=lines
    lists=[]
    for each1 in range(k):
        each1=[]
        lists.append(each1)
    for each in range(len(y)):
        #print(each)
        for r in range(k):
            #print(r)
            if y[each]==r:
                dics[each]=r
                lists[r].append(line[each])


    for result in range(k):
        #path=''
        path=path2+result.__str__()+'.txt'
        with open(path,encoding='utf-8',mode='w') as file:
            for each in lists[result]:
                file.write(each+'\n')
            file.close()
    print(len(lists[0])+len(lists[1]))
    print(dics)
if __name__ == "__main__":

    """
    path = 'C:\\Users\\ruanlu\\Desktop\\test.txt'
    path1='C:\\Users\\ruanlu\\Desktop\\stop_words.txt'
    path2='C:\\Users\\ruanlu\\Desktop\\datas\\result\\'
    """
    path = 'C:\\Users\\ruanlu\\Desktop\\test.txt'
    path1 = 'C:\\Users\\ruanlu\\Desktop\\stop_words.txt'
    path2 = 'C:\\Users\\ruanlu\\Desktop\\datas\\result\\'
    """
    k=8
    stop_words = stop_words(path1)
    tfidf_mat = get_all_vector(path, stop_words)
    #
    myCentroids, clustAssing = kMeans(tfidf_mat, 2, gen_sim, randCent)
    y = kmeans(tfidf_mat, k)
    #y=birch(tfidf_mat,k)
    silhouette_avg, sample_silhouette_values = Silhouette(tfidf_mat, y)  # 轮廓系数
    Draw(silhouette_avg, sample_silhouette_values, y, k,tfidf_mat)
    """
    stop_words = stop_words(path1)
    #tfidf(path,stop_words)
    k = 8
    #stop_words = stop_words(path1)
    #tfidf_mat = get_all_vector(path, stop_words)

    tfidf_mat ,lines= tfidf(path,stop_words)
    #tfidf_mat=PCA(tfidf_mat,1000)
    y = kmeans(tfidf_mat, k)
    sav_ressult(y,path2,k,lines)
    #y=birch(tfidf_mat,k)
    silhouette_avg,sample_silhouette_values = Silhouette(tfidf_mat, y)  # 轮廓系数
    Draw(silhouette_avg, sample_silhouette_values, y, k, tfidf_mat)
