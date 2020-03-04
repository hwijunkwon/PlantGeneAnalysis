from os import listdir, rename
from os.path import isfile, join, isdir
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
#import multiprocessing #추후 멀티프로세싱 도입해야할듯

#이슈사항
#1. 대용량 파일 처리시 multiprocessing 필요
from sklearn.metrics import accuracy_score, classification_report

gene_path =  "./gene"
familyNames = []
Kmer = 10

def makeFamilyFragmentDictionaries():#각 Family별 gene Dictionary 생성
    familyNames.extend( [f for f in listdir(gene_path) if isdir(join(gene_path, f))] )#전역변수 gene_path 폴더안에 있는 폴더 검색
    paths = [join(gene_path, f) for f in familyNames]
    sequenceFragmentForEachFamily = [] #family별 단일 sequence의 fragment set
    dictionariesForEachFamily = [] #family별 전체 fragment dictionary

    for path in paths:#paths에는 family directory folder 경로
        sequences = makeFamilySequences(path)
        dictionary, fragmentForEachSequence = makeFragmentFrequencyDict(sequences)
        sequenceFragmentForEachFamily.append(fragmentForEachSequence)
        dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}
        dictionariesForEachFamily.append(dictionary)
    #print(sequenceFragmentForEachFamily)
    return dictionariesForEachFamily, sequenceFragmentForEachFamily

#makeFamilyDictionaries() 에서 사용
def makeFamilySequences(family_dir_loc):#각 Family별 sequence 리스트 생성
    filenames = [f for f in listdir(family_dir_loc) if isfile(join(family_dir_loc, f))]#각 family 폴더안에 있는 Fasta파일 검색
    sequences = []
    for filename in filenames:
        path = join(family_dir_loc, filename)
        with open(path, "r") as f:
            head, tail = f.read().split('\n', 1)#Fasta format 문서 첫줄제거
            sequence = tail.replace("\n", "")#한줄로
            sequences.append(sequence)
    return sequences

#makeFamilyDictionaries() 에서 사용
def makeFragmentFrequencyDict(sequences):#여러개의 sequence에 있는 fragment를 K의 크기로 잘라서 사전에 추가, 사전에는 fragment의 발생 빈도 기록
    fragmentForEachSequence = []
    dictionary = dict()#사전 생성
    for sequence in sequences:
        length = len(sequence)
        fragments = set()
        for i in range(0, length - Kmer + 1):
            frag = sequence[i: i + Kmer]#k개씩 자름
            fragments.add(frag)

        for fragment in fragments:
            if fragment in dictionary:
                dictionary[fragment] += 1#이미 있으면 +1
            else:
                dictionary[fragment] = 1#없으면 1

        fragmentForEachSequence.append(fragments)
    #print(fragmentForEachSequence)
    return dictionary, fragmentForEachSequence  #dictionary - key: fragment, value: fragment 발생 빈도,  fragmentForEachSequence - [{sequence별 fragment}, {} ...]

def makeOverlappingSeqeunceResult(dictionariesForEachFamily):#패밀리별 fragment를 입력받아 각 fragment가 각각의 패밀리에 속해 있는지 검사
    result = []
    #모든 family의 fragment가 속한 집합(fullSet) 생성
    fullSet = set()
    for family_dict in dictionariesForEachFamily:
        key = family_dict.keys()
        fullSet.update(key)

    #fullSet에 속한 fragment들을 각각의 Family에 속해있는지 검사
    for setItem in fullSet:
        line = setItem
        sum = 0
        for family_dict in dictionariesForEachFamily:
            if setItem in family_dict:
                line += "," + str(1)
                sum += 1
            else:
                line += "," + str(0)
        line += "," + str(sum)
        result.append([line, sum])
    result = sorted(result, key=lambda item: item[1], reverse=True)
    return result

def makeFragmentFrequencyDictToCSV(fragmentFrequencies):
    for i in range(len(fragmentFrequencies)):
        with open(familyNames[i] + "_FragmentFrequencyResult.csv", "w") as f:
            f.write("fragment,frequency\n")
            for key,value in fragmentFrequencies[i].items():
                f.write(key + "," + str(value) +"\n")

def makeOverlappingResultToCSV(overlappingResult):#familyNames: 이름을 나열한 리스트, fragmentFrequency: makeOverlappingSeqeunceResult의 리턴값
    with open("OverlappingResult.csv", "w") as f:
        f.write("fragment")
        for familyName in familyNames:
            f.write("," + familyName)
        f.write(",sum\n")
        for element in overlappingResult:
            f.write(element[0] + "\n")

def convertFragmentToNumber(fragment):#Fragment를 숫자로 변형 A:1, G:2, T:3, C:4
    result = []
    for char in fragment:
        result.append(AGTCSwitcher(char))
    return result

def AGTCSwitcher(char):
    switcher={'A':1, 'G':2, 'T':3, 'C':4,
              "R":5, "Y":6, "K":7, "M":8}
    return switcher[char]

#clustering을 위한 함수
#전처리
# dictForEachFam:makeFamilyFragmentDictionaries() 결과 , frequency: 추출할 유전자 빈도수, 해당 빈도 이상의 fragment를 추출하기 위한 값
def preprocessingForClustering(dictForEachFam):
    x = []
    y = []
    for i in range(len(dictForEachFam)):
        max_frequency = max(dictForEachFam[i].values())
        frequency = max_frequency/2
        print("max_frequency = " + str(max_frequency) + ", frequency: " + str(frequency))

        for key, value in dictForEachFam[i].items():
            #if value >= 2:
            if value >= frequency and value > 1:
                x.append( convertFragmentToNumber(key) )
                y.append(i)
    x = np.array(x) 
    y = np.array(y)
    #print(x)
    #print(y)
    return x,y

#visualization
#PCA
def PCA_2D_Visualization(X, y, figuretitle):
    target_names = familyNames
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    # Percentage of variance explained for each components
    print('explained variance ratio (first two components): %s' % str(pca.explained_variance_ratio_))
    fig = plt.figure()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    lw = 2
    for color, i, target_name in zip(colors, [0,1,2,3,4,5], target_names):
        plt.subplot(2,3,i+1)
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
        plt.title(target_name)

    fig.suptitle(figuretitle)
    fig.tight_layout(pad=2)
    plt.show()

def PCA_3D_Visualization(X, y):
    from sklearn import decomposition
    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)
    # Reorder the labels to have colors matching the cluster results
    y = np.choose(y, [1,2,3,4,5,6,0]).astype(np.float)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.nipy_spectral, edgecolor='k', alpha=0.7)
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()

#t-SNE
def tSNE_Visualization(X,y):
    #Load the iris data
    from sklearn import datasets
    digits = datasets.load_digits()
    # Take the first 500 data points: it's hard to see 1500 points
    X = digits.data[:500]
    y = digits.target[:500]
    #Fit and transform with a TSNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0)
    #Project the data in 2D
    X_2d = tsne.fit_transform(X)
    #visualize the data
    target_ids = range(len(digits.target_names))
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, digits.target_names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()

#clustering
#K-means clustering
def kmeans(x, y, num_cluster):
    # k means 모델 생성
    km = KMeans(n_clusters=num_cluster)
    km.fit(x)
    # training
    y_pred = km.predict(x)
    accuracy = accuracy_score(y, y_pred)
    print("accuracy score:" + str(accuracy))
    print(classification_report(y, y_pred, target_names=familyNames))

    pca = PCA(n_components=2).fit(x)
    pca_2d = pca.transform(x)

    plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, s=50, cmap='viridis', alpha=0.5)
    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

#Hierarchical clustering
def hierarchical(X):
    from scipy.cluster.hierarchy import dendrogram, linkage
    linked = linkage(X, 'single')
    labelList = range(1, 11)
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=labelList,
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()

#visualization
#for i in range(3,21):
Kmer = 11

# fragment 발생 빈도 기록 dictionary 생성
dictionariesForEachFamily, sequenceFragmentForEachFamily = makeFamilyFragmentDictionaries()
makeFragmentFrequencyDictToCSV(dictionariesForEachFamily)

# 패밀리별 fragment를 입력받아 각 fragment가 각각의 패밀리에 속해 있는지 검사 후 결과 출력
result = makeOverlappingSeqeunceResult(dictionariesForEachFamily)
makeOverlappingResultToCSV(result)

# clustering을 위한 전처리
x, y = preprocessingForClustering(dictionariesForEachFamily)

PCA_2D_Visualization(x, y, "PCA, kmer = " + str(Kmer))

#PCA_3D_Visualization(x,y)
#tSNE_Visualization()

#clustering
#kmeans
#kmeans(x,y, 7)
