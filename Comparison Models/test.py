import pandas as pd
import csv
my_set = set()
tot = 0
with open("./data/programmable-web/train.txt", "r",newline='') as f:
    for line in f.readlines():
        line = line.strip()
        apis = line.split(' ')[1:]
        tot+=len(apis)
        for i in apis:
            my_set.add(i)

print(len(my_set))
with open("./data/programmable-web/test.txt", "r",newline='') as f:
    for line in f.readlines():
        line = line.strip()
        apis = line.split(' ')[1:]
        tot += len(apis)
        for i in apis:
            my_set.add(i)
print(len(my_set))
print("总的交互数量为  ",tot)
# for i in my_set:
#     print(i)