# <old version> 
nohup python main.py \
	--rot 1 \
	--data_json <data_json path> \
	--test_txt_pre <prefix of test_txt path> \
	--resume <model path> \
	--gpu_ids 1 > <log path> 2>&1 &

# <new version>
nohup python main.py \
	--rot 1 \
	--new_test \
	--resume <model path> \
	--test_image_txt <test_image_txt path> \
	--src_src <mysql src_src> \
	--test_txt_pre <prefix of test_txt path> > <log path> 2>&1 &

