import re
from nltk.corpus import stopwords
import networkx as nx
stopLex=set(stopwords.words('english'))
import codecs
from sklearn.metrics import accuracy_score
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import svm
#import numpy as np
from sklearn.linear_model import SGDClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
#import codecs
from sklearn.linear_model import RidgeClassifierCV

def loadAuthor(fname,tag,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        if len(review)<2: continue
        reviews.append(review[0].lower())    
        labels.append(tag)
    f.close()

def loadAuthor2(fname,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        #print(review[5])
        if len(review)<2: continue
        reviews.append(review[0].lower())    
        labels.append(review[5].lower())
    f.close()

def loadData(fname,tag,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        if len(review)<2: continue
        reviews.append(review[1].lower())    
        labels.append(tag)
    f.close()
    
def loadAbs2(fname,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        #print(review)
        if len(review)<2: continue
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

def findmax2(vote):
    tmpx=''
    maxv=0
    for x in vote:
        if vote[x]>maxv:
            tmpx=x
            maxv=vote[x]
    return tmpx,maxv

def findmax3(vote):
    tmpx=[]
    maxv=0
    for x in vote:
        if vote[x]>=maxv:
            if vote[x]==maxv:
                tmpx.append(x)
            else:
                tmpx=[]
                tmpx.append(x)
                maxv=vote[x]
    #print('inside',tmpx,maxv)
    if maxv==0: return []
    else: return tmpx

#Load test data
author_test=[]
author_label=[]
loadAuthor('data/cse_test.txt','sigcse',author_test,author_label)
loadAuthor('data/siggraph_test.txt','siggraph',author_test,author_label)
loadAuthor('data/sigir_test.txt','sigir',author_test,author_label)
loadAuthor('data/www_test.txt','www',author_test,author_label)
loadAuthor('data/chi_test.txt','sigchi',author_test,author_label)
loadAuthor('data/cikm_test.txt','cikm',author_test,author_label)
loadAuthor('data/kdd_test.txt','sigkdd',author_test,author_label)
#
loadAuthor2('sample.txt',author_test,author_label)
#print(str(author_test))

#Load author
author = {}

# operation for siga confetence
sigirf=open('data/cse_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['sigcse']=obj.get('sigcse',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

wwwf=open('data/siggraph_train.txt',encoding='utf8',errors='ignore')
for line in wwwf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['siggraph']=obj.get('siggraph',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
wwwf.close()

kddf=open('data/sigir_train.txt',encoding='utf8',errors='ignore')
for line in kddf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['sigir']=obj.get('sigir',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
kddf.close()

sigirf=open('data/www_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['www']=obj.get('www',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

sigirf=open('data/chi_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['sigchi']=obj.get('sigchi',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

sigirf=open('data/cikm_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['cikm']=obj.get('cikm',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

sigirf=open('data/kdd_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['sigkdd']=obj.get('sigkdd',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

#test result
fw=codecs.open('author_test.txt','w',encoding='utf8')
for auth in author:
    #if len(author.get(auth,0))>=4: print(auth+':'+str(author.get(auth,0)))
    fw.write(auth+':'+str(author.get(auth,0))+'\n')

result2 = []
result4 = []
for x in range(len(author_test)):
    autharr=author_test[x].split(':')
    obj={'sigcse':0, 'siggraph':0, 'sigir':0, 'www':0, 'sigchi':0, 'cikm':0, 'sigkdd':0}
    obj2={'sigcse':0, 'siggraph':0, 'sigir':0, 'www':0, 'sigchi':0, 'cikm':0, 'sigkdd':0}
    for auth in autharr:
        auth=re.sub('[^a-z]','',auth.lower())
        if not auth in author.keys(): continue
        authobj=author.get(auth)
        #print(auth,str(authobj))
        for field in authobj:
            obj2[field] += authobj[field]
    #print(obj)
    
    resulttmp2 = findmax(obj2)
    if resulttmp2 == '': result2.append('na')
    else: result2.append(resulttmp2)
    
    resulttmp4=findmax3(obj2)
    #print('outside:',resulttmp4)
    result4.append(resulttmp4)

unk=0
for x in range(len(result2)):
    if result2[x]=='na': 
        author_label[x]='na'
        unk+=1

#print(author_label)
print(str(unk)+'|'+str(len(result2)))

#print(result2)
print(accuracy_score(result2,author_label))

print(result4)    
#print(accuracy_score(result4,author_label))     

reviews=[]
labels=[]
loadData('data/cse_train.txt','sigcse',reviews,labels)
loadData('data/siggraph_train.txt','siggraph',reviews,labels)
loadData('data/sigir_train.txt','sigir',reviews,labels)
loadData('data/www_train.txt','www',reviews,labels)
loadData('data/chi_train.txt','sigchi',reviews,labels)
loadData('data/cikm_train.txt','cikm',reviews,labels)
loadData('data/kdd_train.txt','sigkdd',reviews,labels)


rev_test=[]
labels_test=[]
loadData('data/cse_test.txt','sigcse',rev_test,labels_test)
loadData('data/siggraph_test.txt','siggraph',rev_test,labels_test)
loadData('data/sigir_test.txt','sigir',rev_test,labels_test)
loadData('data/www_test.txt','www',rev_test,labels_test)
loadData('data/chi_test.txt','sigchi',rev_test,labels_test)
loadData('data/cikm_test.txt','cikm',rev_test,labels_test)
loadData('data/kdd_test.txt','sigkdd',rev_test,labels_test)
loadAbs2('sample.txt',rev_test,labels_test)
#print(str(len(rev_test))+'|'+str(len(labels_test)))
#print(labels_test)

counter = CountVectorizer(stop_words='english')
counter.fit(reviews)

#print(counter)
counts_train = counter.transform(reviews)
counts_test = counter.transform(rev_test)

#print(counts_train.shape)
#print(labels)
#print(counts_test)

tfidf_transformer = TfidfTransformer(use_idf=False).fit(counts_train)
X_train_tfidf = tfidf_transformer.fit_transform(counts_train)
X_test_tfidf = tfidf_transformer.fit_transform(counts_test)

clf1 = MultinomialNB()
#clf2 = OneVsRestClassifier(svm.SVC(gamma='scale', decision_function_shape='ovo'))
clf2 = svm.SVC(gamma='scale', decision_function_shape='ovo')
clf3 = svm.LinearSVC(multi_class='ovr', max_iter=5000)
#clf4 = OneVsRestClassifier(MLPClassifier())
clf4 = MLPClassifier()
clf5 = SGDClassifier(n_jobs=7, loss="hinge", penalty="l2", max_iter=5000)
#clf7 = OneVsRestClassifier(XGBClassifier(max_depth=10,colsample_bytree=0.9))
clf7 = XGBClassifier(max_depth=10,colsample_bytree=0.9)
clf8 = RidgeClassifierCV()
clf = [clf1, clf2, clf3, clf4, clf5, clf7, clf8]
#clf = [clf1]
arr = ['MultinomialNB','SVC','LinearSVC','MLPClassifier','SGDClassifier','XGBClassifier','RidgeClassifierCV']
#predictors = [('nb',clf1), ('svc',clf2), ('lsvc',clf3), ('mlp',clf4), ('sgd',clf5), ('xgbc',clf7)]

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
    #print('predict:',pred)
    #print('correct answer:',labels_test)
    s=accuracy_score(pred,labels_test)
    print (s)
    result.append(s)
    if s > maxp:
        maxp = s
        maxx = x
        maxpred = pred

for x in range(len(result)):
    print(arr[x]+':'+str(result[x]))

finalarr=[]
finalvote = []
for x in range(len(rev_test)):
    vote = {'sigcse':0, 'siggraph':0, 'sigir':0, 'www':0, 'sigchi':0, 'cikm':0, 'sigkdd':0}
    for y in range(len(predarr)):
        vote[predarr[y][x]]+=1
    #print(vote)
    tmpresult,tmpval=findmax2(vote)
    finalarr.append(tmpresult)
    finalvote.append(tmpval)

print(finalarr)
print(str(finalvote))
#print(accuracy_score(finalarr,labels_test))
#print('final:',arr[maxx],maxp,maxpred)
out=[]
for x in range(len(result4)):
    if len(result4[x])==0:
        out.append(finalarr[x])
    elif len(result4[x]) > 1:
        if finalarr[x] in result4[x]: 
            out.append(finalarr[x])
        else: out.append(random.choice(result4[x]))
    else:
        out.append(result4[x][0])
        
print(out)
print(str(len(out)))
print(str(len(labels_test)))

print('final predict:',accuracy_score(out,labels_test))

