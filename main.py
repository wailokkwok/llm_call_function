from guidance import models, gen
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import re


def go_to_website(function):
    print('called go_to_website')
    print('variable identified ' + function)
    return 

def gives_product_link(product_name):
    print('called gives_product_link')
    print('variable identified ' + product_name)
    return 


def find_assistance(placeholder):
    print('called find_assistance')
    return 

def complain(placeholder):
    print('called complain')
    return 

def capitalize_first_letter_of_every_word(sentence):
    return sentence.title()


## spacy settings
mistral = models.LlamaCpp('/home/ubuntu/llm/llama-2-13b-chat.Q5_K_M.gguf') 
encoding_model = SentenceTransformer('all-MiniLM-L6-v2')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()



function_dic = {"go to website page":go_to_website,
'provide product link / information':gives_product_link,
"find customer service":find_assistance,
'make complain':complain
}

prompt_dic = {
    "go to website page": 'Answer question in one short single full sentence, identify the page asked in the question and answer in (Target: target identified):\n',
    'provide product link / information': 'Answer question in one short single full sentence, identify the product asked in the question and answer in (Target: target identified):\n',
    "find customer service": 'Answer question in one short single full sentence:\n',
    'make complain': 'Answer question in one short single full sentence:\n'
}


## this is for filtering in NER
function_words = set(word for phrase in function_dic.keys() for word in phrase.lower().split())\

def expand_sentence(noun_verb_dic):
    # Initialize a list to hold all the synonyms
    all_synonyms = []

    # Iterate over the keys and word lists in the input dictionary
    for key, words in noun_verb_dic.items():
        # Set the maximum number of synonyms based on the key
        maximum = 4 if key == 'noun' else 3

        # Iterate over each word in the word list
        for word in words:
            # Get synsets for the word
            pos = wn.NOUN if key == 'noun' else wn.VERB
            synsets = wn.synsets(word, pos=pos)
            # Collect synonyms for the word
            synonyms = set()

            for synset in synsets:
                for lemma in synset.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
                    if len(synonyms) >= maximum:
                        break
                if len(synonyms) >= maximum:
                    break

            # Extend the all_synonyms list with the collected synonyms
            all_synonyms.extend(list(synonyms)[:maximum])

    # Convert the list of all synonyms to a string
    synonym_sentence = ' '.join(all_synonyms)

    return synonym_sentence


# Function to create embeddings from sentences
def create_function_embeddings(function_sentences):
    embeddings = {}
    for function_sentence in function_sentences:
        embedding = encoding_model.encode(function_sentence)
        embeddings[function_sentence] = embedding
    return embeddings

# Function to parse a query and extract nouns and verbs
def get_nouns_verbs(query):
    query_tokens = word_tokenize(query)
    query_tagged_tokens = pos_tag(query_tokens)
    english_stopwords = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    nouns = []
    verbs = []

    for word, pos in query_tagged_tokens:
        if word.lower() not in english_stopwords:
            if pos.startswith('NN'):
                nouns.append(lemmatizer.lemmatize(word.lower(), pos='n'))
            elif pos.startswith('VB'):
                verbs.append(lemmatizer.lemmatize(word.lower(), pos='v'))

    return {'noun':nouns, 'verb':verbs}

def filter_target(output):
    match = re.search(r": ([^)]+)", output)
    if match:
        filtered =  match.group(1).strip()  
        return filtered

    else:
        return None
    


def trigger_function(query, expanded_query, embeddings,expanded_dic):
    query_embedding = encoding_model.encode(expanded_query)
    highest_similarity = -1
    selected_function = None

    # Compare the query embedding with each function embedding
    for function_sentence, func_embedding in embeddings.items():
        similarity = cosine_similarity([query_embedding], [func_embedding])[0][0]
        print(similarity)

        if similarity > highest_similarity:
            highest_similarity = similarity
            selected_function = function_sentence

    # Find the corresponding function from function_dic to call
    if highest_similarity >= 0.3:
        # Map the selected function sentence back to the original function identifier
        if selected_function in expanded_dic:
            return expanded_dic[selected_function]

    return None

query = input("Please enter your query: ")

expanded_dic = {}
for function in function_dic:
    noun_verb_dic = get_nouns_verbs(function)
    output_string = expand_sentence(noun_verb_dic)
    expanded_dic[output_string] = function

function_embeddings = create_function_embeddings(expanded_dic)
query_noun_verb_dic = get_nouns_verbs(query)
print(query)
expanded_query = expand_sentence(query_noun_verb_dic)
call_function = trigger_function(query, expanded_query, function_embeddings,expanded_dic)

if call_function != None:
    prompt = prompt_dic[call_function]
    print(prompt)
    lm = mistral + f'''\
    Q: {prompt + query}
    A: {gen(name="answer", stop="Q:")}'''
    print(lm['answer'])
    variable = filter_target(lm['answer'])
    function_to_call = function_dic[call_function]
    function_to_call(variable)

else:
    lm = mistral + f'''\
    Q: {query}
    A: {gen(name="answer", stop="Q:")}'''
    print(lm['answer'])


        
    

