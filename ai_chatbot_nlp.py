# AI Chatbot with NLP using NLTK

import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Sample Knowledge Base
corpus = '''
Hello, I am an AI chatbot.
I can help you with basic questions.
What is your name?
My name is ChatNLP.
How are you?
I am fine, thank you!
What can you do?
I can answer simple questions.
Tell me a joke.
Why don't scientists trust atoms? Because they make up everything!
Goodbye.
See you later!
'''

# Step 2: Text Preprocessing
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

sent_tokens = nltk.sent_tokenize(corpus)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(str.maketrans('', '', string.punctuation))))

# Step 3: Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["Hi there!", "Hello!", "Hey!", "Hi! How can I help you?"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Step 4: Response Generation
def generate_response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = vectorizer.fit_transform(sent_tokens)
    val = cosine_similarity(tfidf[-1], tfidf)
    idx = val.argsort()[0][-2]
    flat = val.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I'm sorry, I don't understand."
    else:
        robo_response = sent_tokens[idx]
    sent_tokens.pop()
    return robo_response

# Step 5: Chat Loop
print("AI Chatbot is ready! Type 'bye' to exit.")
while True:
    user_input = input("You: ")
    user_input = user_input.lower()
    if user_input == 'bye':
        print("Bot: Goodbye! Have a great day!")
        break
    elif greeting(user_input):
        print("Bot:", greeting(user_input))
    else:
        print("Bot:", generate_response(user_input))
