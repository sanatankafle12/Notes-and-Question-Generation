Type a paragraph of text to generate a Note and MCQ based on Extractive Summarization and T5 Transformers.

T5 base transformer from Hugging face was used. It was Fine tuned to generate a question on the basis of context and answer provided
Documentation of the model
https://huggingface.co/google-t5/t5-base

SQUAD Dataset was used which was downloaded from:
https://rajpurkar.github.io/SQuAD-explorer/

The rouge value for questions were observed as below: 
<p align="center">
  <img src="https://github.com/sanatankafle12/Notes-and-Question-Generation/assets/42962016/1c97f1ab-a4e7-4c0e-a9db-316b1b814e0d" width="400" height="300">
</p>
The average of which was around 0.66.
For the Summary part, TDIDF and Text Rank has been Implemented.

TF-IDF is  used to identify important terms or keywords in the document. These important terms can then be used as a basis for extracting or selecting the most relevant sentences or phrases from the document to form a summary.

Text-rank On the other hand calculates similarity between each sentences using cosine similarity and the sentence that has the highest value among each other is displayed on the top

## Demo

https://github.com/sanatankafle12/major/assets/42962016/69dbc507-dc38-47f9-967e-0110b404b6ae



## Requirements
``` pip install -r requirements.txt```

 ## Running Locally

### Using Notebooks:
1. [generator/train.ipynb](generator/train.ipynb) - trains the model save it locally. 
2. [generator/graph.ipynb](generator/graph.ipynb) - Generates multiple choice questions using trained data. 

### Run web version using Django
1.  Run the 1st step shown in Using Notebook
2.  ```python manage.py migrate```
3.  ```python manage.py runserver```

The project was done as a part of final Year Project.
