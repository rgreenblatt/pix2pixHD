################################ Testing ################################
# first precompute and cluster all features
python3 encode_features.py --name label2city_1024p_feat --netG local --ngf 32 --resize_or_crop none;
# use instance-wise features
python3 test.py --name label2city_1024p_feat ---netG local --ngf 32 --resize_or_crop none --instance_feat