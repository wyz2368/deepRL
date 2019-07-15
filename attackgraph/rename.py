import os
import re


path = os.getcwd() + '/name_processing'
files = os.listdir(path)

print(path)
for file in files:
    if 'DS_Store' in file:
        continue
    idx = int(re.findall('\d+', file)[0])
    print(file,idx)
    os.rename(os.path.join(path, file), os.path.join(path, 'temp' + str(idx + 83) + '.pkl'))


files = os.listdir(path)

# for file in files:
#     if 'DS_Store' in file:
#         continue
#     idx = int(re.findall('\d+', file)[0])
#     os.rename(os.path.join(path, file), os.path.join(path, 'att_str_epoch' + str(idx) + '.pkl'))

for index, file in enumerate(files):
    if 'DS_Store' in file:
        continue
    idx = int(re.findall('\d+', file)[0])
    os.rename(os.path.join(path, file), os.path.join(path, 'def_str_epoch' + str(idx) + '.pkl'))