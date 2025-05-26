## Training on RSDD-SAR and Custom Dataset
The following guide explains how to implement Sparse R-CNN-OBB with a custom dataset.  
While the setup is demonstrated using the RSDD-SAR dataset, the overall process can be followed in the same way for other datasets.

### 1. Ground-truth Conversion
To train and evaluate on the RSDD-SAR dataset, convert the annotations to COCO format by running [convert_RSDD_to_COCO_Detectron.py](./convert_RSDD_to_COCO_Detectron.py).  
Set the `DATASET_phase` parameter to `train` or `"test"` as needed.  
The script will generate the required JSON files if successful.


- [RSDD_train_inshore_COCO_OBB_Detectron.json](./RSDD_train_inshore_COCO_OBB_Detectron.json) 
- [RSDD_test_COCO_OBB_Detectron.json](./RSDD_train_inshore_COCO_OBB_Detectron.json) 
- [RSDD_test_inshore_COCO_OBB_Detectron.json](./RSDD_train_inshore_COCO_OBB_Detectron.json) 
- [RSDD_test_offshore_COCO_OBB_Detectron.son](./RSDD_train_inshore_COCO_OBB_Detectron.json) 

### 2. Register the Dataset
Modify [builtin_meta.py](./detectron2/data/datasets/builtin_meta.py) to register your dataset.  
In that file, define the `categories` and `instances_meta`, and register them within the `_get_builtin_metadata` function.


### 3. Dataset Mapping
Depending on how your ground-truth boxes and labels are structured, you may need to implement a custom dataset mapper.  
For the RSDD-SAR dataset, an example implementation can be found in [dataset_mapper.py](./projects/Sparse_RCNN_OBB/sparsercnn_obb/dataset_mapper.py) and [rsdd_dataset.py](./projects/Sparse_RCNN_OBB/sparsercnn_obb/rsdd_dataset.py).

### 4. Configuration Files
Set the correct dataset path in [config.py](./projects/Sparse_RCNN_OBB/sparsercnn_obb/config.py), and configure the model properties in [Base-SparseRCNN-OBB.yaml](./projects/Sparse_RCNN_OBB/configs/Base-SparseRCNN-OBB.yaml) and [sparse_rcnn_obb.res50.300pro.RSDD.yaml](./projects/Sparse_RCNN_OBB/configs/sparse_rcnn_obb.res50.300pro.RSDD.yaml).

### 5. Run the Training Script
```    
python projects/Sparse_RCNN_OBB/train_net.py \
 --num-gpus 1 --config-file projects/Sparse_RCNN_OBB/configs/sparse_rcnn_obb.res50.300pro.RSDD.yaml
```
