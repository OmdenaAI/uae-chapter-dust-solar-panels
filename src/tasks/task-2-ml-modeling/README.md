W2.2 Prepare/ Explore ML approaches  
W3.1 Explore the pre-trained network and ML models  
W3.2 CNN training and validation  
W3.3 Performing transfer learning  
W4.1 System testing and accuracy reporting  
   
# Model Finetuning and Results:  

*Goal:  Classification of solar panel images to Clean PV and Dirty PV  
*Classes :  CleanPV, DirtyPV  
*Used a balanced dataset of 880:880 for training.  
*Test Data has a total of 121 samples with 61 in each class  


## Alexnet - 81.25 Accuracy  
  
Notebook : [link](https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/alexnet.ipynb)  
Model Weight: [link](https://drive.google.com/drive/folders/1d_J10h4Q70zJEJEHwJ03iOvhhG9f_beP?usp=sharing)
  
Accuracy Plot(Validation):   
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/Accuracy%20Plot.jpg>  
Confusion Matrix for the TestSet:      
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/Screenshot%202022-09-25%20025622.jpg>    
Classification Report for the TestSet:    
<img src = https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/AlexReport.jpg>    
  
  
Fine Tuning:   
   * Number of epochs trained -100  
   * Batch size – 128  
   * Train Validation split: 80:20  
   * Optimizer- SGD, Learning rate – 0.01, stepsize – 11  
   * Dataset was randomly shuffled  
   * Framework : Tensorflow
    
 ## Restnet18 - 89.77272 Accuracy  
   
 Notebook : [link](https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/resnet_18.ipynb)   
 Model Weight: [link](https://drive.google.com/drive/folders/1kGv3bCGjkxcsVfyzaV38K1JgY4DCP-Jw?usp=sharing)
   
 Accuracy Plot(Validation):   
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/resnetplt.jpg>    
Confusion Matrix for the TestSet:     
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/resnetcon.png>  


Fine Tuning:
   * Number of epochs trained (Earlystopping)-19
   * Batch size -16
   * Train validation split :80-20.
   * Optimizer -SGD, Learning rate -0.001, stepsize -20
   * Dataset was randomly shuffled
   * Earlystopping
   * Test Accuracy  for 122 images is 63.
   * Training loss at epoch 19: 0.22, validation loss: 0.3107  
   * Framework: PyTorch

## Densenet121 - 91.48
  
Notebook : [link](https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/densenet121.ipynb)   
Model Weight: [link](https://drive.google.com/drive/folders/16YuKZPtUzPvpixC3iadRGCfmC-RjJpO-?usp=sharing)
   
Accuracy Plot:   
Training:  
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/training.png>    
Finetuning:  
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/DenseNet.png>  
  
Classification Report for Testset:  
<img src = https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/Densenetreport.png>  

   
Fine Tuning  
   * Number of epochs trained -25(used early stopping and model checkpoints)  
   * Batch size -16  
   * Train validation split :80-20.  
   * Optimizer -SGD, Learning rate -0.001, #stepsize :nb_validation_samples = len(validation_generator.filenames)  
   * validation_steps = int(math.ceil(nb_validation_samples / batch_size))  
   * Dataset was randomly shuffled  
   * Framework: Tensorflow

  
## EfficientNet-B3 - 93.899 Accuracy
  
Notebook : [link](https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/EfficientNet_b3_SolarPanelClassification.ipynb)   
Model Weight: [link](https://drive.google.com/drive/folders/1ic2tMHX5FoftrTjs-9nGvtaIujpDEEml?usp=sharing)  
  
Accuracy Plot for Validation:   
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/Efficientplot.png>    
Confusion Matrix for Testset:     
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/efficientcm.jpg>  
Classification Report of Testset:  
<img src=https://github.com/OmdenaAI/uae-chapter-dust-solar-panels/blob/main/src/tasks/task-2-ml-modeling/Assets/efficientreport.jpg>  
   
   
Fine Tuning  
   * Batch size - 32
   * StratifiedKFold to randomly split into multiple combinations of train/val set
   * Optimizer - ADAM  
   * Learning rate - 0.0003, 
   * Early Stopping, Patience - 5
   * lr_scheduler.ReduceLROnPlateau for reducing learning rate when a metric has stopped improving
   * Dataset was randomly shuffled  
   * Framework: PyTorch
