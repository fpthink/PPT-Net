## Pyramid Point Cloud Transformer for Large-Scale Place Recognition  

by Le Hui, Hang Yang, Mingmei Cheng, Jin Xie, and Jian Yang

### Benchmark Datasets

We use the same benchmark datasets introduced in [PointNetVLAD](https://arxiv.org/abs/1804.03492) for point cloud based place recognition, and they can be downloaded [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx).

* Oxford dataset
* NUS (in-house) Datasets
  * university sector (U.S.)
  * residential area (R.A.)
  * business district (B.D.)

### Project Code

#### Pre-requisites

```
Python 3.6+
Pytorch 1.2
CUDA 10.0
```

#### Dataset set-up

* Download the zip file of the benchmark datasets found [here](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx) and extract the folder. Therefore, you have two folders: 1) benchmark_datasets/ and 2) PPT-Net/

* Generate pickle files: We store the positive and negative point clouds to each anchor on pickle files that are used in our training and evaluation codes. The files only need to be generated once. The generation of these files may take a few minutes.
	```
    cd generating_queries/ 
  
    # For network training! Note that "base_path" should be modified to your path.
    python generate_training_tuples_baseline.py
  
    # For network evaluation! Note that "base_path" should be modified to your path.
    python generate_test_sets.py
  ```

#### Training and Evaluation

* build the ops

  ```
  cd libs/pointops && python setup.py install && cd ../../
  ```

* To train and evaluate PPT-Net, run the following command:

    ```
    # Train & Eval
    # Note that you should change the paths in the yaml file.
    
    sh train.sh pptnet configs/pptnet.yaml
    
    python evaluate.py --config configs/pptnet.yaml --save_path exp/pptnet --model_name train_epoch_29_end.pth
    ```


#### Citation

If you find the code or trained models useful, please consider citing:

```
@inproceedings{hui2021pptnet,
  title={Pyramid Point Cloud Transformer for Large-Scale Place Recognition},
  author={Hui, Le and Yang, Hang and Cheng, Mingmei and Xie, Jin and Yang, Jian},
  booktitle={ICCV},
  year={2021}
}

@article{hui2021epcnet,
  title={Efficient 3{D} Point Cloud Feature Learning for Large-Scale Place Recognition},
  author={Hui, Le and Cheng, Mingmei and Xie, Jin and Yang, Jian and Cheng Ming-Ming},
  journal={Transactions on Image Processing},
  year={2021}
}
```



#### Acknowledgement

Our code refers to [PointNetVLAD](https://github.com/mikacuy/pointnetvlad) and [PointWeb](https://github.com/hszhao/PointWeb).
