import codecs

import jieba
import sys
import re
path='C:\\Users\\ruanlu\\Desktop\\6_21_train_rand.txt'
path1='C:\\Users\\ruanlu\\Desktop\\datas\\neg.txt'
path2='C:\\Users\\ruanlu\\Desktop\\datas\\pos.txt'

def readAndDeal(path):
    path1 =path
    #lists = []
    with open(path1, encoding='utf-8',mode='r') as file:
        str1=""
        str2=""
        pos=[]
        neg=[]
        line=file.readline()
        while line:
            #line.rstrip('\n')
            left=re.split('\t',line)
            #print(left)
            if left[0]=='1':
              pos.append(left[1])
              str1=str1+left[1]
              print(str1)
            else:
                neg.append(left[1])
                str2+=left[1]
            line = file.readline()
        file.close()
        return pos,neg,str1,str2
def writeData(path,lists):
    with open(path,encoding='utf-8',mode='w')as file:
        for each in lists:
            file.write(each)
        file.close()
def read_from_path(path):
    """
        with open(path,mode='r') as file:

        list1=[]
        line=file.readline()
        while line:
            print(line)
            list1.append(jieba.cut(line))
            line=file.readline()
        #words=file.read()
        file.close()
        print(list1)

        words = file.read()
        print(type(words))
        return words
    :param path:
    :return:
    """
    """
    file = codecs.open(path, 'r','utf-8')
    lines = [line.strip() for line in file]
    """
    lines=path
    print(lines[1000])
    str=''
    for each in range(len(lines)):
        print(each)
        str+=lines[each]
    #print(str)
    return str
def stopWords(path):
    words=path
    #words=read_from_path(path)
    result=jieba.cut(words)
    new_word=[]
    for each in result:
        new_word.append(each)
        #print(each)

    return set(new_word)
def del_stop_word(path,stop_word_set):
    mylist= [line.strip() for line in open(stop_word_set).readlines()]
    #print(mylist)
    """
        with open(stop_word_set,'r')as ff:
        line = ff.readline()
        while line:
            mulist.append(line)
            line = ff.readline()
        ff.close()
        """
    path1='C:\\Users\\ruanlu\\Desktop\\datas\\neg1.txt'
    result=stopWords(path)
    #word=jieba.cut(result)
    new_words=[]
    s1=0
    for r in result:
        s=1
        if r not in mylist:

            s1=1
            new_words.append(r)

    with open(path1,encoding='utf-8',mode='w')as f:
        for each in new_words:
            #print(each)
            f.writelines(each)
        f.close()



def opss():
    file = codecs.open(path, 'r','utf-8')
    lines = [line.strip() for line in file]
    with open('C:\\Users\\ruanlu\\Desktop\\test.txt',encoding='utf-8',mode='w')as f:
        for wach in range(9000):
            f.write(lines[wach]+'\n')
        f.close()
if __name__ == '__main__':
    #pos,neg,str1,str2=readAndDeal(path)
    #del_stop_word(str1, 'C:\\Users\\ruanlu\\Desktop\\stop_words.txt')
    #writeData(path1, pos)
    #writeData(path2, neg)
    opss()