## ML4SCI GSOC DeepLense Task 
This GitHub repository contains two Folders, each of which focuses on a different deep learning task. 
## Common Task: Multi-class Classification 
The notebook in Common Task demonstrates a simple image classification task using a convolutional neural network (Transfer Learning). 
The dataset used in this notebook is the one provided for common task, which is a collection of strong lensing image. 
The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance. 
# Results on various Models:  

| Model    | Epochs | Batch Size | Learning Rate | Roc_Auc |
|----------|--------|------------|---------------|----------|
| tf_efficientnet_b4_ns | 10 | 64        | 0.0002        | 0.98    |

## Specific Task V : Lens Finding 
The LensFinding folder has notebook focuses on a more specialized deep learning task, namely identifying gravitational lenses in astronomical images. Detecting gravitational lenses is important for understanding the structure and distribution of matter in the universe. The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance.

# Results on various Models: 

| Model    | Epochs | Batch Size | Learning Rate | Roc_Auc |
|----------|--------|------------|---------------|----------|
| ResNet18 | 30    | 64         | 0.00005         | 0.97   |

## Usage

### 1) Setup
Clone the Repository:
```bash
git clone https://github.com/satheeshbhukya/GSOC-DEEPLENSE.git
cd GSOC-DEEPLENSE
```
- For **Common Task**, use `multiclassification_task.ipynb` notebook.
- For **Lens Finding**, use `G_Lens_detection.ipynb` notebook.

### 2) Hyperparameters Setting
Modify the **CFG** class to change hyperparameters:
```python
class CFG:
    lr_max              = 5e-5
    batch_size          = 64
    size                = [64, 64]
    model_name          = "resnet18"
    target_col          = "target"
    epochs              = 30
    seed                = 42
    weight_decay        = 1e-5
    num_workers         = 2
    max_grad_norm       = 1.0
    pos_weight          = 1.0
    device              = device
    early_stop_patience = 7 
```

### 3) Augmentation
Use `get_transforms` function for data augmentation:
```python

def compute_dataset_stats(df, max_samples=500):
    sample_df = df.sample(min(max_samples, len(df)), random_state=CFG.seed)
    means, stds = [], []
    for path in sample_df['data_path']:
        img = np.load(path).astype(np.float32)
        means.append(img.mean(axis=(1, 2)))
        stds.append(img.std(axis=(1, 2)))
    mean = np.mean(means, axis=0).tolist()
    std  = [max(s, 1e-6) for s in np.mean(stds, axis=0).tolist()]
    return mean, std

DATASET_MEAN, DATASET_STD = compute_dataset_stats(full_train_df)
def get_transforms(*, data):
    norm = A.Normalize(mean=DATASET_MEAN, std=DATASET_STD)
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.size[0], CFG.size[1]),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.CoarseDropout(max_holes=8, max_height=8, max_width=8, p=0.3),
            norm,
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.size[0], CFG.size[1]),
            norm,
            ToTensorV2(),
        ])
```

### 4) Training and Evaluation
Use the `Trainer` class to train and evaluate the model:
```python
if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    trainer    = Trainer(CFG)
    best_model = trainer.train_loop(train_df, valid_df)

    y_true, y_pred, test_score = final_evaluation(best_model, test_df)

    pred_df = pd.DataFrame({
        'actual_target': y_true,
        'pred_proba':    y_pred,
        'pred_target':   (y_pred > 0.5).astype(int)
    })
```

## Dependencies
To run the notebooks, install the following dependencies:
-Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PyTorch
  
## Contact: 
- [satheeshbhukyaa@gmail.com]
