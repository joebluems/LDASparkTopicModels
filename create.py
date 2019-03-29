import random
import string
f = open("TheLongGoodbye", "r")
folder = "./documents/"
count=0
for x in f:
  if len(x)>2:
    if count % 3==0:
      docName = ''.join(random.choice(string.digits + string.ascii_uppercase) for _ in range(8))
      file = open(folder+docName+'.text','w') 

    file.write(x)
    count+=1

    #if count % 3 ==0: file.close()
