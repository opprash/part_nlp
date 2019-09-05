import jieba
import codecs
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
    file = codecs.open(path, 'r','utf-8')
    lines = [line.strip() for line in file]
    str=''.join(lines)
    print(len(lines))
    print('over')
    return str
def stopWords(path,stop_word_set):
    s1=0
    s2=0
    new_words=[]
    mylist = [line.strip() for line in open(stop_word_set).readlines()]
    words=read_from_path(path)
    result=jieba.cut(words.strip())
    print(type(result))
    print(result)
    #new_word=[]
    for each in result:
        print(type(each))
        s1=1
        if each not in mylist:
            s2=1
            new_words.append(each)
            print(len(new_words))
        else:
            s2=0
        if s1==s2:
            print('yes')
        else:
            print('no')
        #new_word.append(each)
        #print(each)
    print("over")
    set(new_words)
    #return set(new_word)
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
    #return new_words
if __name__ == '__main__':

    path='C:\\Users\\ruanlu\\Desktop\\datas\\neg2.txt'
    #path = 'C:\\Users\\ruanlu\\Desktop\\test.txt'
    #del_stop_word(path,'C:\\Users\\ruanlu\\Desktop\\stop_words.txt')
    #read_from_path(path)
    stopWords(path,'C:\\Users\\ruanlu\\Desktop\\stop_words.txt')
