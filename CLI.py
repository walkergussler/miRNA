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
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC as SVM
from sklearn.neural_network import BernoulliRBM, MLPClassifier
import pickle

###############################################################################
#parse files, calculate features
###############################################################################
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
            print('error: empty sequence! abort')
            print(parens)
            sys.exit(seq)
        bpratio=len_bp_ratio(seq,numbp)
        loopLen=centralLoopLen(parens)
        seqlen=len(seq)
        numtails=numTails(parens)
        #x.append([numbp,gc,bpratio,loopLen,energy,numtails])
        x.append([numbp,gc,bpratio,loopLen,energy,seqlen,numtails])
    X=np.array(x)
    return X

###############################################################################
#machine learning part
###############################################################################

def run_kmeans(X,Y):
    prds=kmeans(X)
    pts=score(prds,Y)
    return pts

def run_spec(X,Y):
    prds=spec(X)
    pts=score(prds,Y)
    return pts

def run_dbscan(X,Y):
    prds=dbscan(X)
    pts=score(prds,Y)
    return pts
    
def score(y, predictions):
    count = 0
    for x in range(len(predictions)):
        if predictions[x] == y[x]:
            count += 1
    return float(count)/len(predictions)

def spec(X):
    sc = SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1, n_neighbors=10)
    return sc.fit_predict(X)
    
def dbscan(X):
    db = DBSCAN(eps=.5, min_samples=5, metric='euclidean')
    return db.fit_predict(X)
    
def kmeans(X):
    kmeans = KMeans(n_clusters=2, max_iter=300, random_state=0)
    prds = kmeans.fit_predict(X)
    return prds
    
def run_crbm(X,Y,X_train,Y_train):
    (length,width)=np.shape(X_train)
    with open('kddfile','w') as f:
        f.write(str(width+1)+'\n')
        for q in range(width):
            f.write('@attribute '+str(q)+' numerical\n')
        f.write('@attribute class category 1 2\n@data\n')
        for q in range(length):
            Xlis=[]
            for item in X_train[q]:
                Xlis.append(str(item).strip())
            f.write(' '.join(map(str,Xlis))+' '+str(int(Y_train[q])+1)+'\n')
    (length,width)=np.shape(X)
    with open('test','w') as f:
        f.write(str(width+1)+'\n')
        for q in range(width):
            f.write('@attribute '+str(q)+' numerical\n')
        f.write('@attribute class category 1 2\n@data\n')
        for q in range(length):
            Xlis=[]
            for item in X[q]:
                Xlis.append(str(item).strip())
            f.write(' '.join(map(str,Xlis))+' '+str(int(Y[q])+1)+'\n')
    os.system('./kohonen_kdd 50 100 kddfile test > kdd_result')
    correctline=False
    with open('kdd_result') as f:
        for line in f:
            if correctline==True:
                acc=line.split()[-1]
                return float(acc)
            if 'accuracy statistics' in line:
                correctline=True

def normalize(X,desired_features):
    pca=PCA(n_components=desired_features)
    # pca=PCA(n_components=desired_features,whiten=True)
    return pca.fit_transform(X)

def scale(X):
    return StandardScaler().fit_transform(X)    

def run_mlp(X,Y,X_train,Y_train):
    # model = MLPClassifier()
    model = MLPClassifier(hidden_layer_sizes=(10,10), alpha=0.0001 ,max_iter=1000, activation='tanh')
    model.fit(X_train, Y_train)
    kfold = KFold(n_splits=5,random_state=7)
    results= cross_val_score(model, X, Y, cv=kfold)
    return np.average(results)

def run_rf(X,Y,X_train,Y_train):
    model = RandomForestClassifier(n_estimators=100, max_features=3)
    model.fit(X_train, Y_train)
    kfold = KFold(n_splits=5)
    results= cross_val_score(model, X, Y, cv=kfold)
    return np.average(results)

def run_svm(X,Y,X_train,Y_train):
    model = SVM()
    model.fit(X_train, Y_train)
    kfold = KFold(n_splits=5)
    results= cross_val_score(model, X, Y, cv=kfold)
    return np.average(results)
 
def run_drbm(X,Y,X_train,Y_train):
    rbm = BernoulliRBM(n_components=5, learning_rate=.1, n_iter=200)
    model = MLPClassifier(hidden_layer_sizes=(10,10), alpha=0.0001 ,max_iter=1000, activation='tanh')
    reduced=rbm.fit_transform(X,Y)
    kfold = KFold(n_splits=5)
    results = cross_val_score(model, reduced, Y, cv=kfold)
    return np.average(results)
 
def main(positive,negative,features,scaler):
    print(features)
    print('Extracting positives...')
    positiveX=extract(positive)
    print('Extracting negatives...')
    negativeX=extract(negative)
    print("Done! Building machine learning models...")
    positiveY=np.ones(np.shape(positiveX)[0])
    negativeY=np.zeros(np.shape(negativeX)[0])
    preX=np.concatenate([positiveX,negativeX])
    Y=np.concatenate([positiveY,negativeY])
    if scaler:
        midX=scale(preX)
    else:
        midX=preX
    if features==0:
        X=midX
    else:
        X=normalize(midX,int(features))
    X_train, _, Y_train, _ = train_test_split(X, Y)
    
    rf_acc=run_rf(X,Y,X_train,Y_train)
    svm_acc=run_svm(X,Y,X_train,Y_train)
    crbm_acc=run_crbm(X,Y,X_train,Y_train)
    drbm_acc=run_drbm(X,Y,X_train,Y_train)
    mlp_acc=run_mlp(X,Y,X_train,Y_train)
    kmeans_acc=run_kmeans(X,Y)
    dbscan_acc=run_dbscan(X,Y)
    spec_acc=run_spec(X,Y)
    print('Finished classifying!')
    print('Random Forest accuracy: %.3f' %rf_acc)
    print('KMeans accuracy: %.3f' %kmeans_acc)
    print('CRBM accuracy: %.3f' %crbm_acc)
    print('DRBM accuracy: %.3f' %drbm_acc)
    print('MLP accuracy: %.3f' %mlp_acc)
    print('SVM accuracy: %.3f' %svm_acc)
    print('Spectral clustering accuracy: %.3f' %spec_acc)
    print('DBscan accuracy: %.3f' %dbscan_acc)
    
if __name__=='__main__':
    # main('positives.fas','negatives.fas',0,True)
    import argparse # possible arguments to add: delta, nIter
    parser = argparse.ArgumentParser(description='miRNA prediction software')
    parser.add_argument('-p','--positives', 
        type=str, required=False,default='positives.fas',
        help="The path to the positive examples file")
    parser.add_argument('-n','--negatives', 
        type=str, required=False,default='negatives.fas',
        help="The path to the negative examples file")
    parser.add_argument('-s',  '--scale', 
        action='store_true',  default=True,
        help="Pass this as an argument to scale youtr data.")
    parser.add_argument('-f', '--features', 
        type=int, required=False, default=0,
        help="Number of features to reduce to with PCA")
    
    args = parser.parse_args()
    positives=args.positives
    negatives=args.negatives
    scaler=args.scale
    features=args.features
    main(positives,negatives,features,scaler)
    
# def saveMLP(X,Y):
    # model = MLPClassifier(hidden_layer_size=(10,10), alpha=0.0001, activation='tanh',solver='adam')
    # model.fit(X,Y)
    # return pickle.dumps(model)
    
# def saveRandomForest(X,Y):
    # num_trees = 100
    # max_features = 3
    # test_size = 0.45
    # seed = 7
    # model = RandomForestClassifier(n_estimators=num_trees, max_features=3)
    # model.fit(X,Y)
    # return pickle.dumps(model)
    
# def savekmeans(X):
    # kmeans = KMeans(n_clusters=2, max_iter=300, random_state=0)
    # return kmeans.fit(X)

# def hscore(y, predictions):
    # return metrics.homogeneity_score(y, predictions)
    
# def cscore(y, predictions):
    # return metrics.completeness_score(y, predictions)
    
# def vscore(y, predictions):
    # return metrics.v_measure_score(y, predictions)
        
