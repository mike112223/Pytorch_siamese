# Pytorch Siamese

Both main.py and main1.py are main files. 
Each file can train and test Siamese, the difference between these two files is data preprocessing.

main.py: cv2_loader, sub mean [104.0, 117.0, 123.0]
main1.py: cv2_loader, switch channel BGR to RGB, div 255.0, sub mean=[0.485, 0.456, 0.406], div std=[0.229, 0.224, 0.225]

Specific training instructions refers to train.sh
Specific testing instructions refers to test.sh

folder 'data': store pair_json, data_json, image_json
folder 'log': store training log
folder 'pretrain': store files to produce data
folder 'models': store pytorch models
folder 'test': store test files
folder 'tools': store functional files