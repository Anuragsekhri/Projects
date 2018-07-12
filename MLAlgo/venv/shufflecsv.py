import random
fid = open("classi.csv", "r")
li = fid.readlines()
fid.close()
print(li)

random.shuffle(li)
print(li)

fid = open("classi.csv", "w")
fid.writelines(li)
fid.close()