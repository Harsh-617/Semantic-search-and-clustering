import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
#nltk.download('punkt_tab')
nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


#Downloads the Punkt tokenizer model. 
#word_tokenize() relies on this model to correctly split text into words and punctuation
nltk.download('punkt', quiet=True) 

#Downloads the POS (Part-of-Speech) tagger model.
#pos_tag() uses this to assign grammatical tags like noun, verb, adjective, etc., to each word
nltk.download('averaged_perceptron_tagger', quiet=True) 

#Downloads the list of common English stopwords.
#You filter out non-informative words like “the”, “is”, “and”, etc.
nltk.download('stopwords', quiet=True)

#Downloads the WordNet lexical database.
#WordNetLemmatizer relies on WordNet to know the base form of words (lemmatization).
nltk.download('wordnet', quiet=True)

#-----------------------------------------------------------------------------------------------------------------

#creates an instance of the WordNetLemmatizer class from NLTK
lemmatizer = WordNetLemmatizer()

#Retrieves the list of common English stopwords from NLTK; Converts it to a Python set for fast lookup
stop_words = set(stopwords.words('english'))

#-----------------------------------------------------------------------------------------------------------------

#This function converts POS tags from NLTK’s pos_tag format (Treebank tags) to WordNet format for lemmatization

'''Input: treebank_tag - a POS tag returned by pos_tag().
Examples: 'NN' (noun), 'VB' (verb), 'JJ' (adjective), 'RB' (adverb).

Mapping logic:
treebank_tag.startswith('J') → adjective → wordnet.ADJ ('a')
treebank_tag.startswith('V') → verb → wordnet.VERB ('v')
treebank_tag.startswith('N') → noun → wordnet.NOUN ('n')
treebank_tag.startswith('R') → adverb → wordnet.ADV ('r')
Else → default to noun (wordnet.NOUN)'''

#WordNetLemmatizer requires POS in WordNet format (n, v, a, r) for accurate lemmatization

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
#-----------------------------------------------------------------------------------------------------------------


df = pd.read_csv("data/raw/bbc_news.csv")

#print(df.head())

df['FullText'] = df['title'].astype(str) + " " + df['description'].astype(str)

#print(df['FullText'].head())

#print(df.columns)

df = df.dropna(subset=['FullText']) #drop rows with missing 'fulltext'

#-----------------------------------------------------------------------------------------------------------------

def clean_document(text_string):
    #lower casing
    text_string = text_string.lower()

    #removing punctuation
    text_string = re.sub(r'[^\w\s]', ' ', text_string)

    #cleaning whitespaces after the removal
    text_string = re.sub(r'\s+', ' ', text_string).strip()

    #tokenization
    tokens = word_tokenize(text_string)
    
    #pos tagging
    tagged_tokens = pos_tag(tokens)

    #stopword and numbers filtering
    filtered_tokens = [
        (word, tag)
        for word, tag in tagged_tokens
        if (word not in stop_words) or (tag == 'CD')
    ]

    #pos aware lemmetization
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in filtered_tokens
    ]

    #rejoin tokens
    cleaned_text = " ".join(lemmatized_tokens)
    
    return cleaned_text


# Apply the function to the FullText column
df['CleanedText'] = df['FullText'].apply(clean_document)

# Save the DataFrame to a new CSV in the 'data' folder
df.to_csv("data/cleaned/bbc_news_cleaned.csv", index=False)
