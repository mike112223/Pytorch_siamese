nohup python main.py \
	--data_json data/objectid_to_metadata_undist.json \
	--test_json_pre test/shelf_9_10_11_undist_0605_test \
	--resume models/res50_undist_0613_v0_epoch_0.pth \
	--gpu_ids 1 > log/test_undist_0613_v0_0.log 2>&1 &