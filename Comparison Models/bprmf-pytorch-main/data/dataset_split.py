import math
f = open("dataset.txt", "r", encoding="utf-8")
f1 = open("train.txt", "w", encoding="utf-8")
f2 = open("test.txt", "w", encoding="utf-8")
sum1 = 0
sum2 = 0
for i in f.readlines():
    i=i.strip()
    text = i.split(" ")
    print(text)
    sum = len(text)-1
    test_sum =  1
    train_sum = sum-test_sum
    text1 = []
    text2 = []
    text1.append(text[0])
    text2.append(text[0])
    for data in text[1:test_sum + 1]:
        sum2+=1
        text2.append(data)
    for data in text[test_sum+1:]:
        sum1+=1
        text1.append(data)
    text1 = " ".join(text1)
    text2 = " ".join(text2)
    f1.write(text1+'\n')
    f2.write(text2+'\n')

print("train.txt文件中的交互数量为  ",sum1,"   test.txt文件中的交互数量为  ",sum2)