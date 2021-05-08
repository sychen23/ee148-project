import os

split_info_dir = '/Users/sharon/data/EE148/affordance/DataSplit/'
train_split_info_file = os.path.join(split_info_dir, 'aff_training.txt')
test_split_info_file = os.path.join(split_info_dir, 'aff_testing.txt')
print(train_split_info_file)


data_dir = '/Users/sharon/data/EE148/affordance/Data/'
old_train_directory = os.path.join(data_dir, 'train')
train_directory = os.path.join(data_dir, 'affordance_data/train')
test_directory = os.path.join(data_dir, 'affordance_data/test')

with open(train_split_info_file, 'r') as f:
    for line in f:
        print(line.split()[0])
        img_name = '%s.png' % line.split()[0]
        os.rename('%s/%s' % (old_train_directory, img_name),
                '%s/%s' % (train_directory, img_name))
