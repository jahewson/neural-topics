from ast import literal_eval as make_tuple

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import MWETokenizer


class TextTokenizer:
    
    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

    def run(self, text):
        tokens = word_tokenize(text)
        
        alphaTokens = [t for t in tokens if t.isalpha()]
        lowerTokens = [t if len(t) > 1 and t == t.upper() else t.lower() for t in alphaTokens]
        
        wnl = nltk.WordNetLemmatizer()
        lemmatized = [wnl.lemmatize(t) for t in lowerTokens]
        
        extra_stop_words = ['using', 'proceeding', 'conference', 'international', 'symposium',
                            'workshop', 'IEEE', 'ACM', 'II', 'III', 'IV', 'pp', 'ISBN', 'tex',
                            'USA', 'california', 'january', 'february', 'march', 'april', 'may',
                            'june', 'july', 'august', 'september', 'october', 'november', 'december']
        
        stop_words = [*stopwords.words('english'), *extra_stop_words]
        single_tokens = [t for t in lemmatized if t not in stop_words]
        
        # common multi-word tokens
        mwe_tokenizer = MWETokenizer([
            ('ad', 'hoc'),
            ('vice', 'versa'),
            ('de', 'facto'),
            ('et', 'al'),
            ('de', 'novo')
        ])
        return mwe_tokenizer.tokenize(single_tokens)

    def __call__(self, doc):
        return self.run(doc)


class MultiWordTextTokenizer:
    
    def __init__(self):
        ngrams = []
        with open('bigrams.txt') as fp:
            for line in fp:
                ngrams.append(make_tuple(line))
            
        with open('trigrams.txt') as fp:
            for line in fp:
                 ngrams.append(make_tuple(line))
                    
        self.mwe_tokenizer = MWETokenizer(ngrams)
        self.tokenizer = TextTokenizer()

    def __call__(self, doc):
        word_tokens = self.tokenizer.run(doc)
        return self.mwe_tokenizer.tokenize(word_tokens)
