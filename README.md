# Assignment101
DatAism Assignment

## Your analysis should include the following:
1)	Basic preprocessing / Feature Extraction of signals. Please note down any of the novel preprocessing techniques you can think of that can help improve the results.
2)	Any supervised/unsupervised deep learning modeling (RNN/CNN/Transformers preferred). You can choose the labels based on the modeling approach.
3)	Present results based on the above approach. Please include metrics for evaluating your model performance for e.g. classification report for labeled dataset.
4)	Present 1-2 ideas/ next steps on how you can improve the modeling results.


## Analysis Readme :
The data set chosen was MIT-BIH arrhythmia dataset. The data source for  initial data type and modeling analysis was Kaggle (https://www.kaggle.com/datasets/shayanfazeli/heartbeat). The data was  generated for the paper (https://arxiv.org/pdf/1805.00794.pdf) . For the multimodal approach the data was gathered and processed from Physio-net website.
The  notebooks and the files in the folder:
### Modeling  analysis
- Vanilla_CNN1D.ipynb : This  file contains a basic EDA and a vanilla CNN1D implementation (inspired form a Kaggle notebook).
- Vanilla_CNN1D_with_lstm.ipynb : Added a LSTM layer too see the performance on unbalanced data.   
- Vanilla_CNN1D_with_lstm_balanced.ipynb : Added a LSTM layer too see the performance on balanced data.   
- Vanilla_CNN1D_with_BiLSTM_layer.ipynb : Added a BiLSTM layer too see the performance on balanced data.
- Vanilla_CNN1D_with_BiLSTM_layer_with_attention.ipynb : Added a BiLSTM layer with multiplicative self-attention (https://github.com/CyberZHG/keras-self-attention) to see the performance on balanced data.  
- transformer.ipynb :   a 4 transformer blocks-based transformer encoder


### Multimodal approach (inside multimodal design folder)
Paper : https://ieeexplore.ieee.org/abstract/document/9508527
GitHub Repo :  https://github.com/Vidhiwar/multimodule-ecg-classification

There were lot of problems in the repo. Specially for MIT-BIH data the repo was incomplete.  Still need some time with training the model.  The model architecture consists of multiple modes aggregating into a transformer encoder block.   
To  work with data imbalance replaced the transformer encoder layer a label attention layer (https://aclanthology.org/2022.clinicalnlp-1.2/) 

- age_gender.csv : age and gender distribution 
- datprocess.py : generate STFT(Short Time Fourier transform)   from the NumPy files generated form extract.py (multimodule-ecg-classification/extract.py at master · Vidhiwar/multimodule-ecg-classification · GitHub).
- EDA.ipynb : age distribution.
- female_vec.npy and male_vec.npy : vector representation of male and female keyword generated from BERT instead of Word2vec using the sentence “Arrythmia classification in female and male gender”.
- model.py :  model from the paper.
- model_label_attention.py : replaced the transformer block with  label attention block (https://aclanthology.org/2022.clinicalnlp-1.2/).
- train.ipynb : Training the model from the paper.
- trainingLabel_attent.ipynb: training the modified model. 

## Future ideas : since the size of dataset is very limited the approaches are limited. But we can try approaches like layoutlmv3 on a dataset like this mining amplitude,  time,  frequency and demographic domain  all at the same time. 
