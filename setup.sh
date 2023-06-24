cd det3d/ops/dcn 
python3 setup.py build_ext --inplace

cd .. && cd  iou3d_nms
python3 setup.py build_ext --inplace

cd .. && cd  roiaware_pool3d
python3 setup.py build_ext --inplace

cd .. && cd  pillar_ops
python3 setup.py build_ext --inplace
