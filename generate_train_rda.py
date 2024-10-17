import os

f1=open('train_rda.txt','w')
fs=os.listdir('D:\大四上\FineVib\\network\\be314')
index=0
for item in fs:
    index+=1
    if index%2==0:
        continue
    fname='_'.join(item.split('.')[0].split('_')[:-1])
    f1.write(fname+'\n')