def func(strList):
    newList = []
    for str in strList:
        newStr = str[::-1]
        if(len(newStr) >= 5):
            newList.append(newStr)
    return newList
    
l = ["HelloWorld", "Human", "cat" ,"Banana", "Bone"]
nl = func(l)
print(nl)
