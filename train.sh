nohup python main1.py \
	--train \
	--pair_json data/pair_train_data_shelf_9_10_11_undist_0605.json \
	--data_json data/objectid_to_metadata_undist.json \
	--image_json data/image_data_undist.json \
	--name res50_undist_0614_v1 \
	--batch_size 128 \
	--lr 0.008 > log/train_undist_0614_v1.log 2>&1 &

