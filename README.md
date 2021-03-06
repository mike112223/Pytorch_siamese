# Pytorch Siamese

## 图片预处理
- opencv读图
- 采用选择的策略进行crop
- 固定resize到(224,224)
- 是否对图像进行旋转
- 调整图像通道，从BGR到RGB
- 将每个像素缩放到0~1
- 减均值[0.485, 0.456, 0.406]
- 除以标准差[0.229, 0.224, 0.225]

## 模型说明
- 目前只支持resnet50的训练和测试
- 使用ImageNet的res50模型进行初始化，不使用最后的1000元的分类层，从conv5的avg_pool出来以后单独接三层fc，这三层fc的weight的初始化可以选择默认或者xavier（目前只支持这两种），第二层fc后的relu层也可以根据需求选择关掉或者打开
- 训练和测试，采用两个单独的网络，所以其实这层意义上来说，这个网络不能叫做siamese

## 训练和测试
- 训练和测试的代码都封装在main.py里面
- 请参考train.sh和test.sh，里面的参数设置请参考main.py