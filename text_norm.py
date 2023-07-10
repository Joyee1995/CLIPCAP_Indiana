import re
import html
import string
import unicodedata
# from nltk.tokenize import word_tokenize

def remove_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))

def remove_non_ascii(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def replace_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_whitespaces(text):
    return text.strip()

# def remove_stopwords(words, stop_words):
#     return [word for word in words if word not in stop_words]

# def stem_words(words):
#     stemmer = PorterStemmer()
#     return [stemmer.stem(word) for word in words]

# def lemmatize_words(words):
#     lemmatizer = WordNetLemmatizer()
#     return [lemmatizer.lemmatize(word) for word in words]

# def lemmatize_verbs(words):
#     lemmatizer = WordNetLemmatizer()
#     return ' '.join([lemmatizer.lemmatize(word, pos='v') for word in words])

# def text2words(text):
#     return word_tokenize(text)

def normalize_text(text):
    text = remove_chars(text)
    text = remove_non_ascii(text)
    text = remove_punctuation(text)
    text = to_lowercase(text)
    text = replace_numbers(text)
    #words = text2words(text)
    #stop_words = stopwords.words('english')
    #words = remove_stopwords(words, stop_words)
    #words = stem_words(words)# Either stem ovocar lemmatize
    #words = lemmatize_words(words)
    #words = lemmatize_verbs(words)
    return text
  
# def normalize_corpus(corpus):
#     return [normalize_text(t) for t in corpus.split()]

if __name__  == '__main__':
    import numpy as np
    import pandas as pd
    csv_fp = "/notebooks/data/chest-xrays-indiana-university/indiana_reports.csv"
    df_report = pd.read_csv(csv_fp)
    df_report = df_report[df_report['impression'].notna()]
    idxs = np.random.choice(df_report.index, 10)
    for idx in idxs:
        impression = df_report.loc[idx, 'impression']
        print("impression_ori:", impression)
        print("impression_norm:", normalize_text(impression))
        print("=====================")
        
        
    