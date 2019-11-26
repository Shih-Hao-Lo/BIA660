import matplotlib.pyplot as plt
import networkx as nx
from networkx import Graph
import re
from nltk.corpus import stopwords
stopLex=set(stopwords.words('english'))

sigirf=open('articles_sigir.txt',encoding='utf8')
sigirabs=[]
sigirG=nx.Graph()
for line in sigirf:
    arr=line.strip().split('\t\t')
    abstract=re.sub('[^a-z]',' ',arr[1].lower()).split(' ')
    tmp=set()
    for word in abstract:
        if word in stopLex: continue
        if len(word)==0: continue
        if word=='na': continue
        tmp.add(word)
    #sigirabs.append(tmp)
    abstract2=arr[1].split('.')
    #print('abstract2'+str(abstract2))
    if len(abstract2)<2: continue
    for line in abstract2:
        realline=re.sub('[^a-z]',' ',line.lower()).split(' ')
        #print('realline'+str(realline))
        for i in range(len(realline)):
            if realline[i] in stopLex or len(realline[i])<=2 or realline[i]=='na':continue
            for j in range(i+1,len(realline)):
                if realline[j] in stopLex or len(realline[j])<=2 or realline[j]=='na':continue
                if not sigirG.has_edge(realline[i],realline[j]):
                    sigirG.add_edge(realline[i],realline[j]) 
                    sigirG[realline[i]][realline[j]]['freq']=1
                else:
                    sigirG[realline[i]][realline[j]]['freq']+=1
sigirf.close()

remove = []
for N1,N2 in sigirG.edges():
    if sigirG[N1][N2]['freq']>=5:print(N1+','+N2+':'+str(sigirG[N1][N2]['freq']))
    if sigirG[N1][N2]['freq']<5:remove.append((N1,N2))
sigirG.remove_edges_from(remove)
#nx.draw(sigirG)
#print('-----------------------------------------------------------')
#for N1,N2 in sigirG.edges():
#    print(N1+','+N2+':'+str(sigirG[N1][N2]['freq']))
cliques=list(nx.find_cliques(sigirG))
sorted_cliques= sorted(cliques, key=len,reverse=True) # sort cliques by size
print (str(sorted_cliques[0]))