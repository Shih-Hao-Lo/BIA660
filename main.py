import re
from nltk.corpus import stopwords
import networkx as nx
stopLex=set(stopwords.words('english'))
import codecs
from sklearn.metrics import accuracy_score
import random

def loadAuthor(fname,tag,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        if len(review)<2: continue
        if review[0]=='na' or len(review[0])==0:
            #print(review)
            continue 
        reviews.append(review[0].lower())    
        labels.append(tag)
    f.close()

def loadAuthor2(fname,reviews,labels):
    f=open(fname,encoding='utf8',errors='ignore')
    for line in f:
        review=line.strip().lower().split('\t')
        #print(review[5])
        if len(review)<2: continue
        if review[0]=='na' or len(review[0])==0: continue 
        reviews.append(review[0].lower())    
        labels.append(review[5].lower())
    f.close()

def processtag(fname,reviews,labels):
    return

def findmax(vote):
    tmpx=''
    maxv=0
    for x in vote:
        if vote[x]>maxv:
            tmpx=x
            maxv=vote[x]
    return tmpx

def findmax2(vote,vote2):
    tmpx=''
    maxv=0
    maxx=-1
    for x in vote:
        if vote[x]>=maxv:
            if maxx == -1:
                tmpx=x
                maxv=vote[x]
                maxx=x
            else:
                if vote[x] == maxv:
                    if vote2[x] > vote2[maxx]:
                        tmpx=x
                        maxv=vote[x]
                        maxx=x
                else:
                    tmpx=x
                    maxv=vote[x]
                    maxx=x
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
    if maxv==0: return ''
    if len(tmpx)>1: return random.choice(tmpx)
    else: return tmpx[0]

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
        obj=author.get(auth,{'tot':0})
        #print('obj:'+str(obj))
        obj['sigcse']=obj.get('sigcse',0)+1
        obj['tot']=obj.get('tot',0)+1
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
        obj=author.get(auth,{'tot':0})
        #print('obj:'+str(obj))
        obj['siggraph']=obj.get('siggraph',0)+1
        obj['tot']=obj.get('tot',0)+1
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
        obj=author.get(auth,{'tot':0})
        #print('obj:'+str(obj))
        obj['sigir']=obj.get('sigir',0)+1
        obj['tot']=obj.get('tot',0)+1
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
        obj=author.get(auth,{'tot':0})
        #print('obj:'+str(obj))
        obj['www']=obj.get('www',0)+1
        obj['tot']=obj.get('tot',0)+1
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
        obj=author.get(auth,{'tot':0})
        #print('obj:'+str(obj))
        obj['sigchi']=obj.get('sigchi',0)+1
        obj['tot']=obj.get('tot',0)+1
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
        obj=author.get(auth,{'tot':0})
        #print('obj:'+str(obj))
        obj['cikm']=obj.get('cikm',0)+1
        obj['tot']=obj.get('tot',0)+1
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
        obj=author.get(auth,{'tot':1})
        #print('obj:'+str(obj))
        obj['sigkdd']=obj.get('sigkdd',0)+1
        obj['tot']=obj.get('tot',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

#test result
fw=codecs.open('author_test.txt','w',encoding='utf8')
for auth in author:
    if len(author.get(auth,0))>=4: print(auth+':'+str(author.get(auth,0)))
    fw.write(auth+':'+str(author.get(auth,0))+'\n')

result = []
result2 = []
result3 = []
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
            if field=='tot': continue
            if obj[field]==0: obj[field]= authobj[field]/authobj['tot']
            else: obj[field] = obj[field] * (authobj[field]/authobj['tot'])
            obj2[field] += authobj[field]
    #print(obj)
    resulttmp = findmax(obj)
    if resulttmp == '': result.append('na')
    else: result.append(resulttmp)
    
    resulttmp2 = findmax(obj2)
    if resulttmp2 == '': result2.append('na')
    else: result2.append(resulttmp2)
    
    resulttmp3,tmpv3 = findmax2(obj,obj2)
    if resulttmp3 == '' or tmpv3==0: result3.append('na')
    else: result3.append(resulttmp3)
    
    resulttmp4=findmax3(obj2)
    #print('outside:',resulttmp4)
    if resulttmp4 == '': result4.append('na')
    else: result4.append(resulttmp4)


for x in range(len(result)):
    if result[x]=='na': author_label[x]='na'

#print(author_label)

#print(result)
print(accuracy_score(result,author_label))

#print(result2)
print(accuracy_score(result2,author_label))

#print(result3)
print(accuracy_score(result3,author_label))

#print(result4)    
print(accuracy_score(result4,author_label))        