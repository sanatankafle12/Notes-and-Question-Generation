The project was done as a part of final Year Project.<br>
Type a paragraph of text to generate a Note and MCQ based on Extractive Summarization and T5 Transformers.<br>
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
