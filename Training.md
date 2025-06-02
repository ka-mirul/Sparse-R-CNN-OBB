## Training on RSDD-SAR and Custom Dataset
This guide shows how to use Sparse R-CNN-OBB with a custom dataset, using [RSDD-SAR](https://github.com/makabakasu/RSDD-SAR-OPEN) as an example. The same steps apply to other datasets.
Make sure the installation is complete and verified by following the steps in the [Installation Guide](./README.md) before proceeding.



### 1. Ground-truth Conversion
The RSDD-SAR dataset can be downloaded [here](https://github.com/makabakasu/RSDD-SAR-OPEN). \
To train and evaluate using RSDD-SAR dataset, convert the ground truth annotations to COCO format by running [convert_RSDD_to_COCO_Detectron.py](./convert_RSDD_to_COCO_Detectron.py).  
Set the `DATASET_phase` parameter to `"test"`, `"test_inshore"`, or `"test_offshore"` as needed. The script will generate the required JSON files upon success.
- [RSDD_test_COCO_OBB_Detectron.json](./RSDD_test_COCO_OBB_Detectron.json) 
- [RSDD_test_inshore_COCO_OBB_Detectron.json](./RSDD_test_inshore_COCO_OBB_Detectron.json) 
- [RSDD_test_offshore_COCO_OBB_Detectron.json](./RSDD_test_offshore_COCO_OBB_Detectron.json) 

### 2. Register the Dataset
Modify [builtin_meta.py](./detectron2/data/datasets/builtin_meta.py) to register your dataset.  
In that file, define the `categories` and `instances_meta`, and register them within the `_get_builtin_metadata` function.


### 3. Dataset Mapping
Depending on how your ground-truth boxes and labels are structured, you may need to implement a custom dataset mapper.  
For the RSDD-SAR dataset, an example implementation can be found in [dataset_mapper.py](./projects/Sparse_RCNN_OBB/sparsercnn_obb/dataset_mapper.py) and [rsdd_dataset.py](./projects/Sparse_RCNN_OBB/sparsercnn_obb/rsdd_dataset.py).

### 4. Configuration Files
Set the correct dataset path in [config.py](./projects/Sparse_RCNN_OBB/sparsercnn_obb/config.py), and configure the model and training properties in [Base-SparseRCNN-OBB.yaml](./projects/Sparse_RCNN_OBB/configs/Base-SparseRCNN-OBB.yaml) and [sparse_rcnn_obb.res50.300pro.RSDD.yaml](./projects/Sparse_RCNN_OBB/configs/sparse_rcnn_obb.res50.300pro.RSDD.yaml).

### 5. Run the Training Script
```    
python projects/Sparse_RCNN_OBB/train_net.py \
 --num-gpus 1 --config-file projects/Sparse_RCNN_OBB/configs/sparse_rcnn_obb.res50.300pro.RSDD.yaml
```
Once training is finished, the trained model will be saved at `output/model_final.pth`.
