# python3 -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h_no_iou.py --work_dir ./work_dirs/configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h_no_iou_1_7 --seed 0 --gpus 4
#python3 -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h.py --checkpoint work_dirs/configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h_1_7/epoch_20.pth --work_dir ./work_dirs/configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h_1_7


#python3 -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h.py --checkpoint ckpt/pillarnet_full_6194.pth --work_dir pillarnet_full_6194
python3 -m torch.distributed.launch --nproc_per_node=4 ./tools/dist_test.py configs/nusc/pillarnet/nusc_centerpoint_pillarnet_h.py --checkpoint ckpt/pillarnet_full_6709.pth --work_dir pillarnet_full_6709
