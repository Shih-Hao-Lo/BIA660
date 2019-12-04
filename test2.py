from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import codecs
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifierCV

def loadData(fname,tag,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        if len(review)<2: continue
        if review[1]=='na' or len(review[1])==0:
            #print(review)
            continue 
        reviews.append(review[1].lower())    
        labels.append(tag)
    f.close()
    
def loadAbs2(fname,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        #print(review)
        if len(review)<2: continue
        if review[1]=='na' or len(review[1])==0: continue 
        reviews.append(review[1].lower())    
        labels.append(review[5].lower())
    f.close()

def findmax(vote):
    tmpx=''
    maxv=0
    for x in vote:
        if vote[x]>maxv:
            tmpx=x
            maxv=vote[x]
    return tmpx

reviews=[]
labels=[]
loadData('cse_train.txt','sigcse',reviews,labels)
loadData('siggraph_train.txt','siggraph',reviews,labels)
loadData('sigir_train.txt','sigir',reviews,labels)
loadData('www_train.txt','www',reviews,labels)
loadData('chi_train.txt','sigchi',reviews,labels)
loadData('cikm_train.txt','cikm',reviews,labels)
loadData('kdd_train.txt','sigkdd',reviews,labels)


rev_test=[]
labels_test=[]
loadData('cse_test.txt','sigcse',rev_test,labels_test)
loadData('siggraph_test.txt','siggraph',rev_test,labels_test)
loadData('sigir_test.txt','sigir',rev_test,labels_test)
loadData('www_test.txt','www',rev_test,labels_test)
loadData('chi_test.txt','sigchi',rev_test,labels_test)
loadData('cikm_test.txt','cikm',rev_test,labels_test)
loadData('kdd_test.txt','sigkdd',rev_test,labels_test)
loadAbs2('sample.txt',rev_test,labels_test)

#print(labels_test)
counter = CountVectorizer(stop_words='english')
counter.fit(reviews)

#print(counter)
counts_train = counter.transform(reviews)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data

print(counts_train.shape)
#print(labels)
#print(counts_test)

tfidf_transformer  = TfidfTransformer(use_idf=False).fit(counts_train)
X_train_tfidf = tfidf_transformer.fit_transform(counts_train)
X_test_tfidf = tfidf_transformer.fit_transform(counts_test)

clf1 = MultinomialNB()
clf2 = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf3 = svm.LinearSVC(multi_class='ovr', max_iter=3000)
clf4 = MLPClassifier()
clf5 = SGDClassifier(n_jobs=7, loss="hinge", penalty="l2", max_iter=3000)
#clf6 = KNeighborsClassifier(n_neighbors=3)
clf7 = XGBClassifier(max_depth=10,colsample_bytree=0.9)
#clf7 = XGBClassifier(learning_rate =0.01,n_estimators=5000,max_depth=4,min_child_weight=6,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.005,objective= 'binary:logistic',nthread=4,scale_pos_weight=1,seed=27) 
clf8 = RidgeClassifierCV()
clf = [clf1, clf2, clf3, clf4, clf5, clf7, clf8]
arr = ['MultinomialNB','SVC','LinearSVC','MLPClassifier','SGDClassifier','XGBClassifier','RidgeClassifierCV']
#predictors = [('nb',clf1), ('svc',clf2), ('lsvc',clf3), ('mlp',clf4), ('sgd',clf5), ('xgbc',clf7)]

fw=codecs.open('vote_test.txt','w',encoding='utf8')

result=[]
maxpred = []
maxp = 0;
maxx = 0;
predarr = []
for x in range(len(clf)):
    clf[x].fit(counts_train,labels)
    pred=clf[x].predict(counts_test)
    predarr.append(pred)
    print(arr[x])
    print('predict:',pred)
    #fw.write(str(pred)+'\n')
    #print('correct answer:',labels_test)
    s=accuracy_score(pred,labels_test)
    #print (s)
    result.append(s)
    if s > maxp:
        maxp = s
        maxx = x
        maxpred = pred

fw.write(str(predarr))
fw.close()

for x in range(len(result)):
    print(arr[x]+':'+str(result[x]))

finalarr=[]

for x in range(len(rev_test)):
    vote = {'sigcse':0, 'siggraph':0, 'sigir':0, 'www':0, 'sigchi':0, 'cikm':0, 'sigkdd':0}
    for y in range(len(predarr)):
        vote[predarr[y][x]]+=1
    #print(vote)
    tmpresult=findmax(vote)
    finalarr.append(tmpresult)

print(finalarr)
print(accuracy_score(finalarr,labels_test))
print('final:',arr[maxx],maxp,maxpred)
    
        
        