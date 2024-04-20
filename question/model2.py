import nltk
import math
import re
#from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from .preprocessing import *
from rouge import Rouge
#import numpy as np


def Formula_identification(text):
    formula_patterns = (
    r'((y|x)\s*=\s*.+),',  # Example: y = mx + b
    r'if\s(.+)',  # Example: if condition
    r'({.+)'
    r'([A-Za-z]+\s*=\s*[A-Za-z]+\s*[*]\s*[A-Za-z]+)',  # Example: M = A * B
    r'([A-Za-z]+\s*[*]\s*[A-Za-z]+)',  # Example: M * A
    r'([A-Za-z]+\s*=\s*[A-Za-z]+\s*[/]\s*[A-Za-z]+)',  # Example: D = A / B
    r'([A-Za-z]+\s*[/]\s*[A-Za-z]+)',  # Example: D / A
    r'([A-Za-z]+\s*=\s*[A-Za-z]+\s*[+]\s*[A-Za-z]+)',  # Example: C = A + B
    r'([A-Za-z]+\s*[+]\s*[A-Za-z]+)',  # Example: C + A
    r'([A-Za-z]+\s*=\s*[A-Za-z]+\s*[-]\s*[A-Za-z]+)',  # Example: E = A - B
    r'([A-Za-z]+\s*[-]\s*[A-Za-z]+)',  # Example: E - A
    r'([A-Za-z]+\s*=\s*[A-Za-z]+\s*[*]{2}\s*[A-Za-z]+)',  # Example: F = A ** B
    r'([A-Za-z]+\s*[*]{2}\s*[A-Za-z]+)',  # Example: F ** A
    r'([A-Za-z]+\s*=\s*[A-Za-z]+\s*[\^]\s*[A-Za-z]+)',  # Example: G = A ^ B
    r'([A-Za-z]+\s*[\^]\s*[A-Za-z]+)'  # Example: G ^ A
    )
    sentences = text.split('.')

    formulas = []
    for x in sentences:
        for pattern in formula_patterns:
            if re.search(pattern, x):
                formulas.append(re.search(pattern, x).group(1))
                break
    return(formulas)


def n_grams(text):
    words = nltk.word_tokenize(text)
    n =3
    ngrams = list(nltk.ngrams(words, n))
    count_dict = {}

    for item in ngrams:
        if item in count_dict:
            count_dict[item] += 1
        else:
            count_dict[item] = 1
    fixed_dict = {str(key): value for key, value in count_dict.items()}
    return(fixed_dict)
        

def tfidf(text, stopWords):
    frequency_matrix = {}
    for sent in text:
        freq_table = {}
        words=sent.split()
        for word in words:
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:10]] = freq_table

    tf_matrix = {}
    for sentence, value in frequency_matrix.items():
        tf_table = {}
        sentence_word_count = len(value)
        for word, count in value.items():
            tf_table[word] = count/sentence_word_count
        tf_matrix[sentence] = tf_table

    words_in_doc = {}
    for sent,f_table in frequency_matrix.items():
        for word, count in f_table.items():
            if word in words_in_doc:
                words_in_doc[word]+=1
            else:
                words_in_doc[word] = 1

    idf_matrix = {}
    for sent, f_table in frequency_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(len(text)/float(words_in_doc[word]))
        idf_matrix[sent] =idf_table

    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table

    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        if count_words_in_sentence !=0:
            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
        else:
            sentenceValue[sent]=0

    sumValues = 0

    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    average = (sumValues / len(sentenceValue))
    threshold = average
    sentence_count = 0
    summary = []

    for sentence in text:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary.append(sentence)
            sentence_count += 1
    return(summary)


def cosine_similarity(a, b):
    dot_product = sum(i * j for i, j in zip(a, b))
    norm_a = math.sqrt(sum(i * i for i in a))
    norm_b = math.sqrt(sum(j * j for j in b))
    return dot_product / (norm_a * norm_b)


def line(text,stopWords,sentence):
    rouge = Rouge()
    text, unique_words, words, sentences = preprocess_text(text)
    summary,plot,dictionary = text_rank(text)
    summary_tfidf = tfidf(sentence, stopWords)
    summary = '. '.join(summary)
    summary_tfidf = '. '.join(summary_tfidf)
    scores_text_rank = rouge.get_scores([summary], [text])
    scores_tfidf = rouge.get_scores([summary_tfidf], [text])
    metrics = ['rouge-1', 'rouge-2', 'rouge-l']
    text_rank_scores = scores_text_rank[0]
    tfidf_scores = scores_tfidf[0]
    text_rank_rouge1 = text_rank_scores['rouge-1']['f']
    text_rank_rouge2 = text_rank_scores['rouge-2']['f']
    text_rank_rougel = text_rank_scores['rouge-l']['f']
    tfidf_rouge1 = tfidf_scores['rouge-1']['f']
    tfidf_rouge2 = tfidf_scores['rouge-2']['f']
    tfidf_rougel = tfidf_scores['rouge-l']['f']
    plt.plot(metrics, [text_rank_rouge1, text_rank_rouge2, text_rank_rougel], label='TextRank', marker='o')
    plt.plot(metrics, [tfidf_rouge1, tfidf_rouge2, tfidf_rougel], label='TF-IDF', marker='o')
    plt.xlabel('ROUGE Metrics')
    plt.ylabel('F1-Score')
    plt.title('Comparison of ROUGE Scores: TextRank vs. TF-IDF')
    plt.legend()
    return plt


def bar(text,stopWords,sentence):
    rouge = Rouge()
    text, unique_words, words, sentences = preprocess_text(text)
    summary,plot,dictionary = text_rank(text)
    summary_tfidf = tfidf(sentence, stopWords)
    summary = '. '.join(summary)
    summary_tfidf = '. '.join(summary_tfidf)
    scores_text_rank = rouge.get_scores([summary], [text])
    scores_tfidf = rouge.get_scores([summary_tfidf], [text])
    metrics = ['rouge-1', 'rouge-2', 'rouge-l']
    text_rank_scores = scores_text_rank[0]
    tfidf_scores = scores_tfidf[0]
    fig, axs = plt.subplots(len(metrics), figsize=(8, 6))
    for i, metric in enumerate(metrics):
        scores_text_rank = text_rank_scores[metric]['f']
        scores_tfidf = tfidf_scores[metric]['f']
        axs[i].bar(['TextRank', 'TF-IDF'], [scores_text_rank, scores_tfidf])
        axs[i].set_xlabel('Summarization Method')
        axs[i].set_ylabel('F1-Score')
        axs[i].set_title(f'Comparison of {metric} Scores')
        for j, score in enumerate([scores_text_rank, scores_tfidf]):
            axs[i].text(j, score, str(round(score, 2)), ha='center', va='bottom')
    plt.tight_layout()
    return fig


def is_a_relationship(text):
    grams = n_grams(text)
    list_ = ['is', 'is a', 'type', 'kind', 'category', 'class', 'group', 'species', 'example', 'instance', 'form', 'variety', 'model', 'variation', 'subtype', 'subcategory', 'subclass', 'subgroup', 'subspecies', 'subexample', 'subinstance', 'subform']
    nodes = ()
    edges = ()
    for x in grams:
        for j in x:
            if j in list_:
                nodes = nodes + (x[0], x[2], )
                edges = edges + ([x[0], x[2], {'relationship': 'is a'}], )
    return(nodes,edges)
 
def part_of_relationship(text):
    grams = n_grams(text)
    relationship_list = ['part of', 'component of', 'element of', 'member of', 'subpart of', 'subcomponent of', 'subelement of', 'submember of']
    nodes = ()
    edges = ()
    for x in grams:
        for j in x:
            if j in relationship_list:
                nodes = nodes + (x[0], x[2], )
                edges = edges + ([x[0], x[2], {'relationship': 'part of'}], )
    return(nodes,edges)


def semantic_net(text):
    nodes_all = tuple()
    edges_all = tuple()
    nodes,edges = is_a_relationship(text)
    nodes_all += (nodes, )
    edges_all += (edges, )
    semantic_net = nx.Graph()
    semantic_net.add_nodes_from(nodes_all)
    for x in edges_all:
        semantic_net.add_nodes_from(tuple(x))
    plt.figure()
    pos = nx.spring_layout(semantic_net)
    nx.draw_networkx(semantic_net, pos, with_labels=True, node_color="lightblue", font_size=10, node_size=1000)
    edge_labels = nx.get_edge_attributes(semantic_net, "relationship")
    nx.draw_networkx_edge_labels(semantic_net, pos, edge_labels=edge_labels)
    return(plt)


def text_rank(text):
    sentence_graph = nx.Graph()
    sentences = text.split('. ')
    words = list(set(text.split()))
    adjacency_matrix = [[0 for x in range(len(sentences))] for y in range(len(sentences))]
    scores = [0 for x in range(len(sentences))]
    length = len(sentences)
    
    sentences_embedding = {}
    value = 0
    for x in sentences:
        sentences_embedding[x] = value
        sentence_graph.add_node(value)
        value+=1

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                a = [1 if word in sentences[i] else 0 for word in words]
                b = [1 if word in sentences[j] else 0 for word in words]
                similarity = cosine_similarity(a, b)
                adjacency_matrix[i][j] = similarity
                sentence_graph.add_edge(sentences_embedding[sentences[i]] , sentences_embedding[sentences[j]] ,relation = similarity)


    for i in range(len(sentences)):
        sum_adjacency = sum(adjacency_matrix[i])
        score = sum_adjacency / len(sentences)
        scores[i] = score

    sorted_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    summarized = []
    for x in range(int(length)):
        summarized.append(sorted_sentences[x][0])
    plt.figure(figsize=(8, 10)) 
    pos = nx.spring_layout(sentence_graph) 
    edge_labels = nx.get_edge_attributes(sentence_graph, 'relation')
    edge_labels_rounded = {k: round(v, 2) for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        sentence_graph,
        pos,
        edge_labels=edge_labels_rounded,
        font_color='red'
    )
    label_y_offset = 0.06  # Offset for label position along y-axis
    for node, (sentence, score) in enumerate(sorted_sentences):
        plt.text(
            pos[node][0],
            pos[node][1] + label_y_offset,
            f"Score: {score:.2f}",
            ha='center',
            va='bottom'
        )
    nx.draw(sentence_graph, pos, with_labels=True, node_color='lightblue', node_size=800, edge_color='gray')
    plt.axis('off')
    return(summarized,plt,sentences_embedding)
