# Dataset

## A. CALVIN

### 📥 A.0 Download

To download the CALVIN task_D_D dataset:
```bash
cd $DIWALI_ROOT/dataset
sh download.sh calvin
```

### ⚗️ A.1. Preprocess CALVIN Play Data for World Model Training

To have a faster dataloading and training, we preprocess the raw CALVIN dataset by  
- extracting only the essential fields
- resizing pixel observations to `64 x 64`
```bash
python scripts/dataset/preprocess_calvin_dataset.py
```

### ⛏️ A.2. Extract Expert Data for Diffusion Policy Training
After featurizing the whole CALVIN dataset (i.e., after 1.2 [here](../README.md#12-featurizer)), extract featurized expert demonstrations from the play data with:
```bash
python scripts/dataset/extract_expert_demos_feat.py
```

You can extract expert demonstrations with privilege state and/or images, run:
```bash
python scripts/dataset/extract_expert_demos_state.py
(Optional) python scripts/dataset/extract_expert_demos_img.py
```
Note: Please use the same seed across all the above extraction runs to index the exact same episodes.

### 🪄 A.3. Generate Class-Balanced Data for Reward Classifier Training
For Step 3, we first generate task-completion reward labels by rolling out expert demos in the env and then curate the featurized expert demonstrations to have equal class-distribution since data points with task-completion rewards are comparatively low. To get this, we run the following in order:
```bash
python scripts/rewcls/generate_rewards_calvin.py
python scripts/rewcls/create_rewcls_dataset.py
python scripts/rewcls/balance_rewcls_dataset.py
```

**Note:** Each of these scripts are linked to a separate config file. Please familirize with them before running these scripts.
