# Can you Judge a Book by it's Cover?
Deep Learning approach on PyTorch to whether you can you judge a book by it's title and cover. 

# Introduction and Problem Statement
The common idiom states that one can't judge a book by it's cover. This project seeks to try and do so using Deep Learning Techniques on PyTorch. Hence, the problem statement is  formulated as "Given an image or the title of a book, is it possible to classify the book into the correct genre?"

The learning goals in this project was:
- Implement an end-to-end framework for image classification from scratch using a custom dataset
- Try and learn how to implement the pre-trained [BERT classifier](https://github.com/google-research/bert), and learn implementation techniques in NLP.

# Dataset
For this project, I found the [Uchida Book Dataset](https://github.com/uchidalab/book-dataset), which has data of about 57,000 books taken from Amazon. This dataset contains book cover images, title, author, and subcategories for each respective book. Each of the books are classified into 30 classes.

An important aspect of the dataset to be noted is that a book on Amazon can have multiple genres associated with this. However, when creating the dataset, the authors randomly chose one class out of the many classes that a book may be associated with. Hence, it would be prudent to also use Top 3 and Top 5 percent accuracy when comparing the results of the network because in reality a book may be in classes other than it's assigned classes. 

This project is divided into two parts:
1. Classification based on title using BERT
2. Classification based on cover image using ResNet

For the second task, I used a subset of the dataset due to computational constraints. In specific, I used only books from 10 classes to make predictions. Since the CSV file with the title was considerably smaller in size, we use all 30 classes for the title prediction model.  

## Classificication Based on Cover Image
For this portion of the project, I used a pre-trained networks and modified the output layer such that it outputs only 10 classes. I referred to [this tutorial on the PyTorch website](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) to load a custom dataset, and implement the project. The network was trained for 25 epochs with the Adam optimizer. A comparison of the results are shown below.

The results are shown in the table below:
**Model**|**Top 1 Accuracy %**|**Top 3 Accuracy %**|**Top 5 Accuracy %**
:-----:|:-----:|:-----:|:-----:
ResNet50|41.424|71.614|87.1875
ResNet101| | | 
VGG-19| | | 
DenseNet121| | | 

## Classification Based on Book Title
For this, we use the entire dataset, i.e. we classify amongst all the 30 classes in the dataset. I used this [helpful tutorial for spam classification](https://www.analyticsvidhya.com/blog/2020/07/transfer-learning-for-nlp-fine-tuning-bert-for-text-classification/) and accordingly modified it for the problem at hand. From the results below, we see that the BERT base model gains considerably good accuracy after training for 50 epochs with the AdamW optimizer.  

**Model**|**Top 1 Accuracy %**|**Top 3 Accuracy %**|**Top 5 Accuracy %**
:-----:|:-----:|:-----:|:-----:
BERT|44.982|68.193|78.491

# Future Work
Try and create a combined method that takes uses both the title and the cover image and then uses both information to make a combined prediction of the image. 

# Citations and Resources
1.  [B. K. Iwana, S. T. Raza Rizvi, S. Ahmed, A. Dengel, and S. Uchida, "Judging a Book by its Cover," arXiv preprint arXiv:1610.09204 (2016).](https://arxiv.org/abs/1610.09204)
2.  [BERT from Google AI Blog](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html)
