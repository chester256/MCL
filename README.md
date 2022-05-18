## Multi-level Consistency Learning for Semi-supervised Domain Adaptation
Zizheng Yan, Yushuang Wu, Guanbin Li, Yipeng Qin, Xiaoguang Han, Shuguang Cui, "Multi-level Consistency Learning for Semi-supervised Domain Adaptation", IJCAI 2022. [[paper](https://arxiv.org/pdf/2205.04066)]


### 1. Requirements
```shell
pip install -r requirements.txt
```
The project is developed with PyTorch and POT. It should work with different versions.

### 2. Data Preparation
For Domainnet, please follow [MME](https://github.com/VisionLearningGroup/SSDA_MME) to prepare the data. The expected dataset path pattern is like `your-domainnet-data-root/domain-name/class-name/images.png`.

For Office-Home, please download the [resized images](https://drive.google.com/file/d/1OkkrggGq35QSZNPuYhmrdmMZXtkqnBqO/view?usp=sharing) and extract, you will get a .pkl and a .npy file, then specify their paths at line 205 in `loader/office_home.py`.  As the dataset scale of office-home is small, we resize the images to 256x256 and save to a single array so that the data loading is faster.

### 3. Training

Specify the dataset paths and domains in `train.sh`, and

```shell
bash train.sh
```

### 4. Acknowledgement

The code is partly based on [MME](https://github.com/VisionLearningGroup/SSDA_MME)

### 5. Citation

```
@article{yan2022multi,
  title={Multi-level consistency learning for semi-supervised domain adaptation},
  author={Yan, Zizheng and Wu, Yushuang and Li, Guanbin and Qin, Yipeng and Han, Xiaoguang and Cui, Shuguang},
  journal={IJCAI},
  year={2022}
}
```