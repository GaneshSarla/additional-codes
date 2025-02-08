import pandas as pd
import re
import nltk
import gensim
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.models.phrases import Phrases, Phraser
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load Dataset (Replace with actual file path)
df = pd.read_csv("BBS.csv")  # Ensure the dataset has a column 'text' for reviews

def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Lemmatization & Stopword removal
    return tokens

# Apply preprocessing
df['processed_reviews'] = df['text'].astype(str).apply(preprocess_text)

# Create bigram and trigram models
bigram = Phrases(df['processed_reviews'], min_count=5, threshold=100)
bigram_mod = Phraser(bigram)
trigram = Phrases(bigram[df['processed_reviews']], threshold=100)
trigram_mod = Phraser(trigram)

# Apply bigrams and trigrams
df['processed_reviews'] = df['processed_reviews'].apply(lambda x: trigram_mod[bigram_mod[x]])

# Create Dictionary and Corpus
dictionary = corpora.Dictionary(df['processed_reviews'])
dictionary.filter_extremes(no_below=5, no_above=0.5)  # Remove rare and overly common words
corpus = [dictionary.doc2bow(text) for text in df['processed_reviews']]

# Optimize number of topics using coherence score
def compute_coherence_values(dictionary, corpus, texts, start=5, limit=15, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=50)
        model_list.append(model)
        coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherence_model.get_coherence())
    return model_list, coherence_values

model_list, coherence_values = compute_coherence_values(dictionary, corpus, df['processed_reviews'])

# Plot coherence score
def plot_coherence(coherence_values, start=5, limit=15, step=1):
    x = range(start, limit, step)
    plt.figure(figsize=(10, 5))
    plt.plot(x, coherence_values, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("Optimal Number of Topics Based on Coherence Score")
    plt.grid()
    plt.show()

plot_coherence(coherence_values)

# Select best model based on coherence score
best_model_index = coherence_values.index(max(coherence_values))
lda_model = model_list[best_model_index]

# Assign dominant topic to each review
def get_dominant_topic(bow):
    topic_probs = lda_model.get_document_topics(bow)
    if topic_probs:
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
        return f"Topic {dominant_topic+1}"
    return "No Topic"

df['Topic'] = df['processed_reviews'].apply(lambda x: get_dominant_topic(dictionary.doc2bow(x)))

# Save results to Excel
df[['text', 'Topic']].to_excel("Reviews_with_Topics.xlsx", index=False)
print("Reviews with assigned topics saved successfully to Reviews_with_Topics.xlsx!")

# Display discovered topics
for i, topic in lda_model.print_topics():
    print(f"Topic {i+1}: {topic}")
