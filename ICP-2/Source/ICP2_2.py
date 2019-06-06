def string_alternative():
    str= input("Enter the string")
    str1 = ""
    for i in range(len(str)):
        if(i%2==0):
            str1=str1+str[i]
    print(str1)
string_alternative()