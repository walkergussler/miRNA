from SimpleXMLRPCServer import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler
import numpy as np
import sys, os, subprocess
from tempfile import NamedTemporaryFile
from scipy.cluster.hierarchy import dendrogram, ward
import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering as AGG
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier

# Restrict to a particular path.
class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)

# Create server
server = SimpleXMLRPCServer(("localhost", 8001), requestHandler=RequestHandler)

print('Connected!')

#Define auxilary functions
def runRNAfold(file):
    with NamedTemporaryFile(delete=False) as g:
        outname=g.name
        with open(file) as f:
            subprocess.check_call(['RNAfold','--noPS'], stdin=f,stdout=g)
    return outname

def read(file):
    outdic={}
    with open(file) as f:
        lines=f.readlines()
    for line in lines:
        if line[0]=='>':
            pass
        elif line[0] in 'ACGU':
            key=line.strip()
        elif line[0] in '.()':
            (pstring, energy)=figureparens(line.strip())
            outdic[key]=(pstring,energy)
    return outdic

def figureparens(line):
    s=line.split('(')[-1]
    energy=float(s[:-2])
    pstr=line[:-9]
    return(pstr,energy)

def GC_content(seq):
    return (seq.count('C')+seq.count('G'))/float(len(seq))

def num_bp(parens):
    return parens.count('(')+parens.count(')')

def len_bp_ratio(seq,numbp):
    return len(seq)/float(numbp)

def numTails(parens):
    numTails=0
    if parens[0] == "."or parens[-1]== ".":
        numTails+=1
        if parens[0] == "." and parens[-1] == ".":
            numTails+=1
    return numTails

def centralLoopLen(p):
    return len(p.split('(')[-1].split(')')[0])

#supervised learning methods
def randomForest(X,Y):
    num_trees = 100
    max_features = 3
    test_size = 0.45
    seed = 7
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.45,random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=3)
    model.fit(X_train, Y_train)
    predicted = model.predict(X_test)
    #print "Prediction is", predicted
    kfold = KFold(n_splits=5, random_state=7)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=3)
    results = cross_val_score(model, X, Y, cv=kfold)
    return np.average(results)

#unsupervised clustering methods
def kmeans(X):
    kmeans = KMeans(n_clusters=2, max_iter=300, random_state=0)
    prds = kmeans.fit_predict(X)
    return prds
    
def agg(X):
    agg = AGG(n_clusters=2)
    return agg.fit_predict(X)
    
def dbscan(X):
    db = DBSCAN(eps=.5, min_samples=5, metric='euclidean')
    return db.fit_predict(X)
    
def spec(X):
    sc = SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1, n_neighbors=10)
    return sc.fit_predict(X)

#scoring functions, scaler  
def hscore(y, predictions):
    h = metrics.homogeneity_score(y, predictions)
    return h
    
def cscore(y, predictions):
    c = metrics.completeness_score(y, predictions)
    return c
    
def vscore(y, predictions):
    v = metrics.v_measure_score(y, predictions)
    return v

def score(y, predictions):
    count = 0
    for x in range(len(predictions)):
        if predictions[x] == y[x]:
            count += 1
    return float(count)/len(predictions)

def scale(X):
    return StandardScaler().fit_transform(X)    

def extract(infile):
    foldfile=runRNAfold(infile)
    seqDict=read(foldfile)
    #you have to handle y yourself
    x=[]
    for seq in seqDict:
        (parens,energy)=seqDict[seq]
        numbp=num_bp(parens)
        gc=GC_content(seq)
        if numbp==0:
            print(parens)
            raw_input(seq)
        bpratio=len_bp_ratio(seq,numbp)
        loopLen=centralLoopLen(parens)
        seqlen=len(seq)
        numtails=numTails(parens)
        #x.append([numbp,gc,bpratio,loopLen,energy,numtails])
        x.append([numbp,gc,bpratio,loopLen,energy,seqlen,numtails])
    X=np.array(x)
    return X
    
def main(positive,negative):
    #nameslist = ['numbasepairs', 'gccontent', 'lengthbasepairatio', 'centrallooplength', 'freeenergypernuc', 'seqlen', 'numtails']
    print('Extracting positives...')
    positiveX=extract(positive)
    print('Extracting negatives...')
    negativeX=extract(negative)
    positiveY=np.ones(np.shape(positiveX)[0])
    negativeY=np.zeros(np.shape(negativeX)[0])
    X=np.concatenate([positiveX,negativeX])
    y=np.concatenate([positiveY,negativeY])
    supacc=randomForest(X,y)   
    kmeanspred=kmeans(X)
    unsupacc=score(kmeanspred,y)
    print('finished classifying')
    return(map(float,[supacc,unsupacc]))
        
#you have to register functions called by client
server.register_function(main)

# Run the server's main loop forever, ctrl^c to exit
server.serve_forever()

