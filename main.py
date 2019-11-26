import re
from nltk.corpus import stopwords
import networkx as nx
stopLex=set(stopwords.words('english'))

author = {}

# operation for siga confetence
sigirf=open('articles_sigir.txt',encoding='utf8')
sigirabs=[]
sigirG=nx.Graph()
for line in sigirf:
    arr=line.strip().split('\t\t')
    #print('author:'+arr[0])
    auths=arr[0].split(', ')
    #print(str(auths))
    for auth in auths:
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['sigar']=obj.get('sigar',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
    abstract=re.sub('[^a-z]',' ',arr[1].lower()).split(' ')
    tmp=set()
    for word in abstract:
        if word in stopLex: continue
        if len(word)==0: continue
        if word=='na': continue
        tmp.add(word)
    sigirabs.append(tmp)
    
    for i in range(len(abstract)):
        for j in range(i+1,len(abstract)):
            if not sigirG.has_edge(abstract[i],abstract[j]):
                sigirG.add_edge(abstract[i],abstract[j]) 
                sigirG[abstract[i]][abstract[j]]['freq']=1
            else:
                sigirG[abstract[i]][abstract[j]]['freq']+=1
sigirf.close()

wwwf=open('articles_www.txt',encoding='utf8')
wwwabs=[]
for line in wwwf:
    arr=line.strip().split('\t\t')
    #print('author:'+arr[0])
    auths=arr[0].split(', ')
    for auth in auths:
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['www']=obj.get('www',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    abstract=re.sub('[^a-z]',' ',arr[1].lower()).split(' ')
    tmp=set()
    for word in abstract:
        if word in stopLex: continue
        if len(word)==0: continue
        if word=='na': continue
        tmp.add(word)
    wwwabs.append(tmp)
wwwf.close()

kddf=open('articles_kdd.txt',encoding='utf8')
kddabs=[]
for line in kddf:
    arr=line.strip().split('\t\t')
    #print('author:'+arr[0])
    auths=arr[0].split(', ')
    for auth in auths:
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['kdd']=obj.get('kdd',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    abstract=re.sub('[^a-z]',' ',arr[1].lower()).split(' ')
    tmp=set()
    for word in abstract:
        if word in stopLex: continue
        if len(word)==0: continue
        if word=='na': continue
        tmp.add(word)
    kddabs.append(tmp)
kddf.close()

#test result
#for auth in author:
#sa    print(auth+':'+str(author.get(auth,0)))
#print(str(kddabs))
