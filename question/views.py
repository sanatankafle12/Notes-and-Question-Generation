from django.shortcuts import render
from django.http import HttpResponse
from .forms import TextEntryForm
#from .summarized import summarize
from .model2 import tfidf,text_rank
from .preprocessing import preprocess_text
from .fill_in_the_blanks import get_fill_in_the_blanks
import random
from .keyword import get_keywords
import json
#from .generator1 import get_options
from .question_gen import question_g
from .better import get_distract

stopWords = ['ever', 'under', 'although', 'eight', 'many', 'toward', 'would', 'thru', 'her', 'thereby', 'in', 'meanwhile', 'per', 'seeming', 'whereupon', 'anywhere', 'empty', 'then', 'there', 'here', 'twelve', 'my', 'nowhere', 'some', 'ourselves', '‘ll', 'itself', 'only', 'seemed', 'these', 'such', 'much', 'less', 'ten', 'hence', 'this', 'as', 'also', 'wherever', 'while', 'done', 'moreover', 'three', 'than', 'becomes', 'of', 'yourself', 'were', 'nothing', 'an', 'nor', 'enough', 'his', '’re', 'does', 'they', 'even', 'behind', 'may', 'take', 'afterwards', 'have', 'for', 'formerly', 'something', 'now', 'put', 'ours', 'eleven', 'none', 'out', 'besides', 'again', 'hers', 'first', 'via', 'anyhow', 'latter', 'its', 'whereby', 'hundred', 'say', 'hereby', 'not', 'with', 'often', 'a', 'before', 'but', 'each', 'becoming', 'full', 'from', 'within', 'both', 'below', 'others', 'show', 'whenever', 'too', 'mostly', 'anyway', 'mine', 'once', 'yourselves', 'hereafter', 'another', 'is', 'serious', 'few', 'together', 'might', 'go', 'n’t', 'into', 'whole', 'keep', 'thereafter', 'to', 'whither', 'how', 'further', 'otherwise', '’ll', 'due', 'fifteen', 'whether', 'sixty', 'always', 'amount', 'without', 'where', 'myself', 'who', 'using', 'by', 'made', 'should', 'what', 'nine', 'must', 'indeed', 'being', 'do', 'almost', 'up', 'hereupon', 'namely', 'however', 'amongst', 'it', 'most', 'off', 'your', 'bottom', 'so', 'him', 'perhaps', "'re", 'two', 'seems', 'regarding', 'various', '‘re', 'became', 'are', 'did', 'be', 'thus', 'move', 'and', 'above', 'ca', 'i', 'across', 'all', 'part', 'throughout', 'used', 'six', 'own', 'towards', "'s", 'quite', 'noone', 'them', 'along', "'ve", 'nevertheless', 'upon', 'someone', 'third', 'whatever', 'because', 'five', 'had', 'thereupon', "'ll", 'therefore', "'m", 'beforehand', 'please', 'any', 'am', '‘d', 'several', 'cannot', 'on', '’d', 'over', '‘m', 'the', 'us', 'onto', '’m', 'make', 'twenty', 'four', 'latterly', 'next', 'other', 'through', 'when', 'whoever', 'against', 'except', 'everywhere', 'you', 'our', 'me', "'d", '’s', 'during', 'that', '‘s', 'whom', 'if', 'more', 'n‘t', 'yet', 'never', 'was', 'just', 'anyone', 'same', 'top', 'can', 'beside', 'we', 'really', 'herein', 'fifty', 'somehow', 'among', 'she', 'could', 'though', 'beyond', 'else', 'well', 'nobody', 'whence', 'neither', 'until', 'last', 'seem', 'after', 'will', 'has', 'see', 'since', 'sometimes', 'wherein', 'anything', 'least', 'down', 'no', 'whereas', 'herself', 'himself', 'whereafter', 'very', 'been', 'doing', 'between', 'alone', 'everyone', 'still', 'those', 'at', 'thence', 'therein', 'already', '’ve', 'one', 'why', 'get', 'rather', 'former', 'side', 'or', 'every', 'forty', 'he', 'around', 'everything', 'their', 'become', 've', 'which', 'name', 're', 'either', "n't", 'back', 'sometime', 'front', 'call', 'elsewhere', 'whose', 'unless', 'themselves', 'give', 'yours', 'about', 'somewhere']

def process_text(request):
    if request.method == 'POST':
        form = TextEntryForm(request.POST)
        if form.is_valid():
            action = request.POST.get('action')
            text_entry = form.save(commit=False)
            if action == 'notes':
                summary = tfidf(text_entry.text.split('.'),stopWords)
                text, unique_words, words, sentences = preprocess_text(text_entry.text)
                summary2,plot,dictionary = text_rank(text)
                text_entry.save()
                return render(request, 'process_text.html',{'form': form, 'summary': summary,'text_rank':summary2})
            elif action == 'questions':
                question1 = get_fill_in_the_blanks(text_entry.text)
                title = question1['title']
                sentences = question1['sentences']
                keys = question1['keys']
                random.shuffle(keys)
                dictionary = Mcq(text_entry.text)
                questions_json = json.dumps(dictionary)
                return render(request, 'process_text.html',{'form': form,'title':title,'keys':keys,'sentences':sentences,'dictionary':dictionary,"questions_json":questions_json})
    else:
        form = TextEntryForm()
    
    return render(request, 'process_text.html', {'form': form})

def Mcq(text1):
    summarized_text = ". ".join(tfidf(text1.split('.'), stopWords))
    dictionary = []
    keywords = get_keywords(text1,summarized_text)[0]
    questions = question_g(summarized_text,keywords)
    print(keywords)
    print(questions)
    distractors = get_distract(keywords)
    print(distractors)
    for i in range(len(keywords)):
        data_dict = {}
        distractors[i].append(keywords[i])
        random.shuffle(distractors[i])
        data_dict = {
            'keyword': keywords[i],
            'question': questions[i],
            'distractors': distractors[i]
        }
        dictionary.append(data_dict)
    return(dictionary)


