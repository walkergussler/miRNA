import sys
import xmlrpclib

def main(positive,negative):
    print('Connecting...')
    s=xmlrpclib.ServerProxy('http://localhost:8001')
    [supacc,unsupacc]=s.main(positive,negative)
    print('Supervised methods accuracy: %.3f' % float(supacc))
    print('Unsupervised methods accuracy: %.3f' % float(unsupacc))    
    sys.exit('operation completed successfully!')

if __name__=='__main__':
    if len(sys.argv)!=3:
        sys.exit('usage: python client.py <positives> <negatives>')
    else:
        main(sys.argv[1],sys.argv[2])
