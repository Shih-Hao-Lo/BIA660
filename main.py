import re
from nltk.corpus import stopwords
stopLex=set(stopwords.words('english'))
from sklearn.metrics import accuracy_score
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
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
    if maxv==0: return []
    else: return tmpx

print('Loading Data')
#Load test data
author_test=[]
author_label=[]

#Add the file path of test data here and line 289
loadAuthor2('sample.txt',author_test,author_label)

#Load author
author = {}

# operation for siga confetence
sigirf=open('data/cse_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigcse']=obj.get('sigcse',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data2/sigcse_training.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigcse']=obj.get('sigcse',0)+1
        author[auth]=obj
sigirf.close()
wwwf=open('data/siggraph_train.txt',encoding='utf8',errors='ignore')
for line in wwwf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['siggraph']=obj.get('siggraph',0)+1
        author[auth]=obj
wwwf.close()
wwwf=open('data2/siggraph_training.txt',encoding='utf8',errors='ignore')
for line in wwwf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['siggraph']=obj.get('siggraph',0)+1
        author[auth]=obj
wwwf.close()
kddf=open('data/sigir_train.txt',encoding='utf8',errors='ignore')
for line in kddf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigir']=obj.get('sigir',0)+1
        author[auth]=obj
kddf.close()
kddf=open('data2/sigir_training.txt',encoding='utf8',errors='ignore')
for line in kddf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigir']=obj.get('sigir',0)+1
        author[auth]=obj
kddf.close()
sigirf=open('data/www_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['www']=obj.get('www',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data2/www_training.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['www']=obj.get('www',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data/chi_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigchi']=obj.get('sigchi',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data2/sigchi_training.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigchi']=obj.get('sigchi',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data/cikm_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['cikm']=obj.get('cikm',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data2/cikm_training.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['cikm']=obj.get('cikm',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data/kdd_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigkdd']=obj.get('sigkdd',0)+1
        author[auth]=obj
sigirf.close()
sigirf=open('data2/sigkdd_training.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        auth=re.sub('[^a-z]','',auth.lower())
        obj=author.get(auth,{})
        obj['sigkdd']=obj.get('sigkdd',0)+1
        author[auth]=obj
sigirf.close()
print('Loading Complete')

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
        for field in authobj:
            obj2[field] += authobj[field]
    
    resulttmp2 = findmax(obj2)
    if resulttmp2 == '': result2.append('na')
    else: result2.append(resulttmp2)
    
    resulttmp4=findmax3(obj2)
    result4.append(resulttmp4)

for x in range(len(result2)):
    if result2[x]=='na': 
        author_label[x]='na'

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
#Add the file path of test data here and line 89
loadAbs2('sample.txt',rev_test,labels_test)

print('CountVectorizer Working...')
counter = CountVectorizer(stop_words='english')
counter.fit(reviews)

counts_train = counter.transform(reviews)
counts_test = counter.transform(rev_test)

tfidf_transformer = TfidfTransformer(use_idf=False).fit(counts_train)
X_train_tfidf = tfidf_transformer.fit_transform(counts_train)
X_test_tfidf = tfidf_transformer.fit_transform(counts_test)
print('CountVectorizer Complete...')
print('Prepare to Fit...')
clf1 = MultinomialNB()
clf2 = OneVsRestClassifier(svm.SVC(gamma=0.001, decision_function_shape='ovo'))
clf3 = svm.LinearSVC(multi_class='ovr', max_iter=5000)
clf4 = OneVsRestClassifier(MLPClassifier())
clf5 = SGDClassifier(n_jobs=7, loss="hinge", penalty="l2", max_iter=5000)
clf7 = OneVsRestClassifier(XGBClassifier(max_depth=10,colsample_bytree=0.9))
clf8 = RidgeClassifierCV()
clf = [clf1, clf2, clf3, clf4, clf5, clf7, clf8]

arr = ['MultinomialNB','SVC','LinearSVC','MLPClassifier','SGDClassifier','XGBClassifier','RidgeClassifierCV']

result=[]
maxpred = []
maxp = 0;
maxx = 0;
predarr = []
for x in range(len(clf)):
    print(arr[x]+' training...')
    clf[x].fit(counts_train,labels)
    pred=clf[x].predict(counts_test)
    predarr.append(pred)
    s=accuracy_score(pred,labels_test)
    result.append(s)
    print(arr[x]+' complete...')
    if s > maxp:
        maxp = s
        maxx = x
        maxpred = pred

print('Fit Complte...')
finalarr=[]
finalvote = []
for x in range(len(rev_test)):
    vote = {'sigcse':0, 'siggraph':0, 'sigir':0, 'www':0, 'sigchi':0, 'cikm':0, 'sigkdd':0}
    for y in range(len(predarr)):
        vote[predarr[y][x]]+=1
    tmpresult,tmpval=findmax2(vote)
    finalarr.append(tmpresult)
    finalvote.append(tmpval)

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

print('Predict Result:')
print(out)
print('final Accuracy:',accuracy_score(out,labels_test))

