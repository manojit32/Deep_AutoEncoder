import numpy as np
def run1(k,y):
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    cluster1 = KMeans(n_clusters=5)  
    #cln=AgglomerativeClustering(n_clusters=5)
    cluster1.fit(k)
    correct = 0
    pred=[]
    X=k
    cluster=[[],[],[],[],[]]
    for i in range(len(X)):
        toPredict = np.array(X[i].astype(float))
        toPredict = toPredict.reshape(-1, len(toPredict))
        prediction = cluster1.predict(toPredict)
        #print(prediction)
        cluster[prediction[0]].append(i)
        pred.append(prediction)

    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        print(max(score))
        cluster_max.append(clusters[score.index(max(score))])
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])
        
        
def run2(k,y):
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    #cluster1 = KMeans(n_clusters=5)  
    cluster1=GaussianMixture(n_components=5)
    cluster1.fit(k)
    correct = 0
    pred=[]
    X=k
    cluster=[[],[],[],[],[]]
    for i in range(len(X)):
        toPredict = np.array(X[i].astype(float))
        toPredict = toPredict.reshape(-1, len(toPredict))
        prediction = cluster1.predict(toPredict)
        #print(prediction)
        cluster[prediction[0]].append(i)
        pred.append(prediction)

    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        print(max(score))
        cluster_max.append(clusters[score.index(max(score))])
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])
        
        
        
def run3(k,y):
    from sklearn.cluster import KMeans,AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    cln=AgglomerativeClustering(n_clusters=5)
    X=k
    cln.fit(X)
    y1=cln.fit_predict(X)
    for i in range(5):
        print(list(y1).count(i))

    correct = 0
    pred=[]
    cluster=[[],[],[],[],[]]
    for i in range(len(y1)):
        cluster[y1[i]].append(i)
    y_ac=[[],[],[],[],[]]
    for i in range(5):
        print("Number of elements in Cluster "+str(i)+" : "+str(len(cluster[i])))
        for j in cluster[i]:
            y_ac[i].append(y[j])
    probe=[]
    r2l=[]
    u2r=[]
    normal=[]
    dos=[]
    clusters=["probe","r2l","u2r","normal","dos"]
    cluster_max=[]
    for i in range(5):
        print("Cluster "+str(i)+" Purity")
        score=[]
        score.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        probe.append(list(y_ac[i]).count("probe")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        r2l.append(list(y_ac[i]).count("r2l")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        u2r.append(list(y_ac[i]).count("u2r")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        normal.append(list(y_ac[i]).count("normal")/len(y_ac[i]))
        score.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        dos.append(list(y_ac[i]).count("dos")/len(y_ac[i]))
        print(max(score))
        cluster_max.append(clusters[score.index(max(score))])    
        import matplotlib.pyplot as plt
        labels = clusters
        sizes = score
        colors = ['gold', 'yellowgreen', 'red', 'lightskyblue','lightcoral']
        explode = (0.1, 0, 0, 0,0)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.show()
        cluster_max.append(clusters[score.index(max(score))])