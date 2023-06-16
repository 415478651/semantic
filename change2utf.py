import os
import struct,decimal,datetime,itertools
def dbfreader(f):
    numrec,lenheader=struct.unpack('
    numfields=(lenbheader-33)//32
    fields=[]
    for fieldno in xrange(numfields):
        name,typ,size,deci=struct.unpack('<11sc4xBB14x',f.read(32))
        
    ')
'''path="C:\\Users\\ZETTAKIT\\Desktop\\xzq - 副本.txt"
with open(path,encoding='gbk') as f:
    lines = f.readlines()
print(lines)'''