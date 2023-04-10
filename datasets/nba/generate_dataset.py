
from Game import Game
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Process arguments about an NBA game.')

args = parser.parse_args()

data_root = 'datasets/source'
data_target = 'datasets'
if not os.path.exists(data_target):
    os.mkdir(data_target)
json_list = os.listdir(data_root)
print(json_list)
all_trajs = []

for file_name in json_list:
	if '.json' not in file_name:
		continue
	json_path = data_root + '/' + file_name
	game = Game(path_to_json=json_path)
	trajs = game.read_json()   
	trajs = np.unique(trajs,axis=0) 
	print("fgg",trajs.shape)
	all_trajs.append(trajs)

all_trajs = np.concatenate(all_trajs,axis=0)
all_trajs = np.unique(all_trajs,axis=0)
print(len(all_trajs))
index = list(range(len(all_trajs)))
from random import shuffle
shuffle(index)
train_set = all_trajs[index[:int(0.7*len(all_trajs))]]
test_set = all_trajs[index[int(0.7*len(all_trajs)):]]
sanitised_train_set = np.zeros((1,15, 11, 3))

train_set_no_aug = np.zeros((1,15, 11, 3))


for i in range(test_set.shape[0]):
	traj = test_set[i]
	# print(traj[0][0][2])
	if traj[0][0][2] == 0:
		traj = traj.reshape(1,15,11,3)
		sanitised_train_set = np.append(sanitised_train_set,traj,axis = 0)

print(sanitised_train_set.shape)


for i in range(train_set.shape[0]):
	traj = train_set[i]
	# print(traj[0][0][2])
	if traj[0][0][2] == 0:
		traj = traj.reshape(1,15,11,3)
		train_set_no_aug = np.append(train_set_no_aug,traj,axis = 0)

sanitised_train_set = sanitised_train_set[1:,:,:,:]
train_set_no_aug = train_set_no_aug[1:,:,:,:]


print('train num:',train_set.shape[0])
print('test num:',test_set.shape[0])
print('santised', sanitised_train_set.shape[0])
print('no_aug', train_set_no_aug.shape[0])

test_set = sanitised_train_set

np.save(data_target+'/train.npy',train_set)
np.save(data_target+'/test.npy',test_set)
np.save(data_target+'/train_no_aug.npy', train_set_no_aug)