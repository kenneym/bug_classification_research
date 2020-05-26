###########
# Imports #
###########

from bs4 import BeautifulSoup
import spacy
from text_unidecode import unidecode
from word2number import w2n
import en_core_web_sm
import re

#######################
# Preprocessing Steps #
#######################

## Load Spacy
print("Spacy model", "is" if spacy.prefer_gpu() else "is NOT", "using GPU")  
nlp = en_core_web_sm.load()


def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode(text)
    return text




def preprocess_training_text(text, accented_chars=True, 
                       convert_num=False, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=True):
    
    """
    - This function takes strings as input and returns all words in the input text, 
      seperated by spaces.
    - Alter the input options to configure the text preprocessing function.
    """
    

    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()
        
        
    doc = nlp(text) #tokenise text


    clean_text = []
    for token in doc:
        flag = True
        edit = token.text
        # print("Word: ", edit, " Type: ", token.pos_)
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
        # remove punctuations
        if punctuations == True and (token.pos_ == 'PUNCT') and flag == True: 
            flag = False
            
        # remove 'X' characters:
        if token.pos_ == 'X':
            flag = False
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
        # append tokens edited and not removed to list 
        if edit != "" and flag == True:
            clean_text.append(edit)
        
    # Convert back to string:
    new_text = ' '.join(clean_text)
    regex = re.compile('[^a-zA-Z]')
    new_text = regex.sub(' ', new_text)
    words = re.findall(r'\w+.', new_text)
    return ' '.join(words)


# The below function is a variation of the above. You can create as many variations of
# the above function as you would like, simply by changing the parameters.

def preprocess_training_text_with_stops(text, convert=False):
    """ Preproccessing text, but keeps stop words. """
    return preprocess_training_text(text, accented_chars=True,
                       convert_num=False, extra_whitespace=True, 
                       lemmatization=True, lowercase=True, punctuations=True,
                       remove_html=True, remove_num=True, special_chars=True, 
                       stop_words=False)

def text_to_corpus(text, accented_chars=True,
                   convert_num=True, extra_whitespace=True, 
                   lemmatization=True, lowercase=True, punctuations=True,
                   remove_html=True, remove_num=True, special_chars=True, 
                   stop_words=True):
    """
    - This function takes strings as input (may include HTML) and returns all words
      in the input text. This function retains the sentence structure of the input text,
      leaving existing periods wherever they are, replacing all other end punctuation 
      (?, !, ...) with periods, and ending HTML headers and subtitles with periods. This
      function is ideal for converting webpage text to a corpus to be training on a
      word2vec model.
      
    - Alter the input options to configure the text preprocessing function.
    """   
    

    """preprocess text with default option set to true for all steps"""
    if remove_html == True: #remove html tags
        text = strip_html_tags(text)
    if extra_whitespace == True: #remove extra whitespaces
        text = remove_whitespace(text)
    if accented_chars == True: #remove accented characters
        text = remove_accented_chars(text)
    if lowercase == True: #convert all characters to lowercase
        text = text.lower()
        
    # add a period to the end of the text:
    if len(text) > 0 and text[-1] != '.':
        text += '.'
        
    doc = nlp(text) #tokenise text   
    clean_text = []
    
    for token in doc:
        
        flag = True
        edit = token.text
        # print("Word: ", edit, " Type: ", token.pos_)
        
        # remove stop words
        if stop_words == True and token.is_stop and token.pos_ != 'NUM': 
            flag = False
            
        # remove punctuations
        if punctuations == True and (token.pos_ == 'PUNCT' and not token.tag_ == '.') and flag == True: 
            flag = False
            
        # remove 'X' characters:
        if token.pos_ == 'X':
            flag = False
        
        # remove special characters
        if special_chars == True and token.pos_ == 'SYM' and flag == True: 
            flag = False
            
        # remove numbers
        if remove_num == True and (token.pos_ == 'NUM' or token.text.isnumeric()) \
        and flag == True:
            flag = False
            
        # convert number words to numeric numbers
        if convert_num == True and token.pos_ == 'NUM' and flag == True:
            edit = w2n.word_to_num(token.text)
            
        # convert tokens to base form
        elif lemmatization == True and token.lemma_ != "-PRON-" and flag == True:
            edit = token.lemma_
            
        # convert all closing punctuation ('.', '!', '?', '...' to periods)
        if token.tag_ == '.' and flag == True:
            clean_text.append('.')
            
        # add text lemmas to the clean text:
        elif edit != "" and flag == True:
            clean_text.append(edit)
            
    return ' '.join(clean_text)



