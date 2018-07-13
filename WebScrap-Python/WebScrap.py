import requests
from bs4 import BeautifulSoup
import threading
import matplotlib.pyplot as plt


dict={};

def func1(lock):
    response = requests.get("https://gaana.com/playlist/gaana-dj-top-hindi-songs-of-70s")
        # print(response.text)
        # print("+++++++++++++++++++++++++++++++++==")
    soup = BeautifulSoup(response.text, "html.parser")

    aTag = soup.find_all(attrs={"data-type": "url"})

    artist = []

    print("-------------------------------")

    for a in aTag[1:]:
        artist.append(a.text)

        # print(artist)

        artdict = {}

        for x in artist:
            if x in artdict:
                artdict[x] += 1
            else:
                artdict[x] = 1

        for x in artdict:
            artdict[x] = artdict[x] / 4

    print(artdict)
    global dict
    with lock:
        dict=artdict


lock = threading.Lock()
t1 = threading.Thread(target = func1, name = "1", args=(lock,))
t1.start()
t1.join()

print("----------------------------------------")
print(dict)

X=[]
Y=[]
X_=[]
i=0

for key in dict.keys():
    X.append(key)
    X_.append(i)
    i=i+1
    Y.append(dict[key])

print(X)
print(Y)
plt.figure(figsize=(15,10))
plt.bar(X_,Y);
plt.xlabel("Singer")
plt.xticks(X_, X,rotation=45,fontsize=9)

plt.ylabel("No of Songs")
plt.title("70's Top 100 Songs")
figManager = plt.get_current_fig_manager()
figManager.full_screen_toggle()
plt.show()

