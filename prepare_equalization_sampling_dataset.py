import os
import json
import random
import jsonlines

type = "equalization"

# please add the dataset file path before run the code 
dataset = '/'

all = [json.loads(line) for line in open(os.path.join(dataset, 'combined.jsonl'))]

random.shuffle(all)

size = len(all)

temp = all[:int(size * 0.80)]
test = all[int(size * 0.80):]

size_temp = len(temp)
train = temp[:int(size_temp * 0.90)]
dev = temp[int(size_temp * 0.90):]

if type == 'equalization':
  # Do equalization sampling in train set
  equalized_train = []
  num_samples = 2000

  categories = ['nationality', 'disabs', 'class_', 'asian', 'others', 'women', 'white', 'lgbt', 'religion', 'black']

  category_samples = {}
  for i in categories:
    category_samples[i] = []

  # add all non-hate as is, without equalization
  for i in train:
    if i['label'] == 0:
      equalized_train.append(i)

  for i in categories:
    for j in train:
      if j['label'] == 1:
        if i == j['category'][0]:
          category_samples[i].append(j)

  for i in categories:
    samples = category_samples[i]

    if len(samples) == 0:
      continue

    if len(samples) >= num_samples:
      equalized_train += samples[:num_samples]

    else:
      equalized_train += samples # put all existing samples 
      current_samples_num = len(samples) # randomly pick rest of the smaples
      for j in range(num_samples - current_samples_num):
        random_sample = samples[random.randint(0, len(samples) - 1)]
        equalized_train.append(random_sample)


  # please add the store path before run the code
  storePath = "/"

  with jsonlines.Writer(open(storePath + 'train.jsonl', 'w')) as writer:
    writer.write_all(equalized_train)
  
  with jsonlines.Writer(open(storePath + 'test.jsonl', 'w')) as writer:
    writer.write_all(test)
  
  with jsonlines.Writer(open(storePath + 'dev.jsonl', 'w')) as writer:
    writer.write_all(dev)

else:

  # please add the store path before run the code
  storePath = "/"
  
  with jsonlines.Writer(open(storePath + 'train.jsonl', 'w')) as writer:
    writer.write_all(train)

  with jsonlines.Writer(open(storePath + 'test.jsonl', 'w')) as writer:
    writer.write_all(test)

  with jsonlines.Writer(open(storePath + 'dev.jsonl', 'w')) as writer:
    writer.write_all(dev)



