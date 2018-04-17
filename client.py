import sys
import xmlrpclib

def main(positive,negative,unknown,method):
    #print('Connecting...')
    s=xmlrpclib.ServerProxy('http://localhost:8001',allow_none=True)
    ret=s.main(positive,negative,unknown,method)
    if method==None and unknown==None:
        print('Supervised methods accuracy: %.3f' % float(ret[0]))
        print('Unsupervised methods accuracy: %.3f' % float(ret[1]))
        sys.exit('operation completed successfully!')
    else:
        for item in ret:
            print(item)
        sys.exit()

if __name__=='__main__':
    if len(sys.argv)not in [3,5]:
        print('usage: python client.py <positive> <negative>')
        print('usage: python client.py <positive> <negative> <unknown> <method>')
        print('usage: python client.py <positive> <negative> <unknown> <method> > <output>')
        sys.exit("method can be 'rf' for random forest (supervised) or 'km' for kmeans (unsupervised)")
    elif len(sys.argv)==3:
        main(sys.argv[1],sys.argv[2],None,None)
    elif len(sys.argv)==5:
        main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])