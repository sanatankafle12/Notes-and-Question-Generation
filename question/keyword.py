import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from sense2vec import Sense2Vec

import string
import pke
import traceback
s2v = Sense2Vec().from_disk('/mnt/c/Users/sanatan/OneDrive/Desktop/Question-MCQ-_Answer_Generation-main/UI/MCQ/question/s2v_old')

def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1,
                                      threshold=0.75,
                                      method='average')
        keyphrases = extractor.get_n_best(n=15)
        

        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()
    return out

from flashtext import KeywordProcessor


def get_keywords(originaltext,summarytext):
  keywords = get_nouns_multipartite(originaltext)
  keyword_processor = KeywordProcessor()
  for keyword in keywords:
    keyword_processor.add_keyword(keyword)

  keywords_found = keyword_processor.extract_keywords(summarytext)
  keywords_found = list(set(keywords_found))

  important_keywords =[]
  for keyword in keywords:
    sense = s2v.get_best_sense(keyword)
    if keyword in keywords_found and sense !=None:
      important_keywords.append(keyword)

  return (important_keywords[:4],keywords,keywords_found)

