file = open("C:\\Users\\vidyu\\Desktop\\python code\\abc.txt","r+")
wordcount={}
for word in file.read().split():
    if word.lower() not in wordcount:
        wordcount[word.lower()] = 1
    else:
        wordcount[word.lower()] += 1
print(wordcount)
file.close()
f = open("C:\\Users\\vidyu\\Desktop\\python code\\abc.txt", "w+")
for i in wordcount:
     f.write(i+" : "+str(wordcount[i])+"\n")
f.close()