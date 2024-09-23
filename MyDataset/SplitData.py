import os
import random
import shutil
from itertools import islice

outputFolderPath = "Dataset/SplitData"
inputFolderPath = "Dataset/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

try:
    shutil.rmtree(outputFolderPath)
except OSError as e:
    os.mkdir(outputFolderPath)

os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

listNames = os.listdir(inputFolderPath)
uniqueNames = list(set(name.split('.')[0] for name in listNames))

random.shuffle(uniqueNames)

lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

if lenData != lenTrain + lenTest + lenVal:
    remaining = lenData - (lenTrain + lenTest + lenVal)
    lenTrain += remaining

lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]
print(f'Total Images:{lenData} \nSplit: {len(Output[0])} {len(Output[1])} {len(Output[2])}')

sequence = ['train', 'val', 'test']
for i, out in enumerate(Output):
    for fileName in out:
        shutil.copy(f'{inputFolderPath}/{fileName}.jpg', f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        shutil.copy(f'{inputFolderPath}/{fileName}.txt', f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')

print("Split Process Completed...")

dataYaml = f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'

with open(f"{outputFolderPath}/data.yaml", 'a') as f:
    f.write(dataYaml)

print("Data.yaml file Created...")
