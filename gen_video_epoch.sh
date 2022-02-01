epoch=30
# 30fps * second
length=$(( 30*15 ))

for ((start_frame=0;start_frame<=2000;start_frame=start_frame+100)); 
do
    python3 generate_video.py --name fire_project --dataroot ./datasets/fireworks_dataset/ \
    --fps 30 --ngf 16 --which_epoch $epoch --how_many $length --start_frame $start_frame
done