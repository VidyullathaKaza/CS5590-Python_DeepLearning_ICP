n = int(input("Enter number of elements : "))
list1 = []
list2 = []
for x in range(n):
    list1.append(float(input("Enter "+str(x+1)+" element : ")))
for x in range(n):
    list2.append((list1[x]*0.454))
dictionary = dict(zip(list1, list2))
print(dictionary)