import re
from nltk.corpus import stopwords
import networkx as nx
stopLex=set(stopwords.words('english'))
import codecs

author = {}

# operation for siga confetence
sigirf=open('cse_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['cse']=obj.get('cse',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

wwwf=open('siggraph_train.txt',encoding='utf8',errors='ignore')
for line in wwwf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['siggraph']=obj.get('siggraph',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
wwwf.close()

kddf=open('sigir_train.txt',encoding='utf8',errors='ignore')
for line in kddf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['sigir']=obj.get('sigir',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
kddf.close()

sigirf=open('www_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['www']=obj.get('www',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

sigirf=open('chi_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['chi']=obj.get('chi',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

sigirf=open('cikm_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['cikm']=obj.get('cikm',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

sigirf=open('kdd_train.txt',encoding='utf8',errors='ignore')
for line in sigirf:
    arr=line.strip().split('\t')
    #print('author:'+arr[0])
    auths=arr[0].split(':')
    #print(str(auths))
    for auth in auths:
        if auth.lower() == 'na' or len(auth)==0: continue
        obj=author.get(auth,{})
        #print('obj:'+str(obj))
        obj['kdd']=obj.get('kdd',0)+1
        author[auth]=obj
        #print('obj[sigar]:'+str(obj['sigar']))
    #print(arr)
sigirf.close()

#test result
fw=codecs.open('author_test.txt','w',encoding='utf8')
for auth in author:
    if len(author.get(auth,0))>=4: print(auth+':'+str(author.get(auth,0)))
    fw.write(auth+':'+str(author.get(auth,0))+'\t'+str(len(author.get(auth,0)))+'\n')
