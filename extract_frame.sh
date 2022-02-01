python3 extract_frames.py -video brain.mov -name fireworks_dataset -p2pdir . -width 1280 -height 736

python3 train_video.py --name fire_project --dataroot ./datasets/fireworks_dataset/ --save_epoch_freq 10 --ngf 16 --niter 50 --niter_decay 50

python3 generate_video.py --name fire_project --dataroot ./datasets/fireworks_dataset/ --fps 24 --ngf 16 --which_epoch 50 --how_many 1000