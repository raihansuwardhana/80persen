import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
import streamlit as st

# Download necessary NLTK resource
nltk.download('punkt')

# Download necessary NLTK resource
nltk.download('twitter_samples')

nltk.download('stopwords')

# Explicitly set NLTK data path for Streamlit Share
nltk.data.path.append('/app/nltk_data')

# Fungsi untuk preprocessing cleansing
def preprocess(text):
    # Menghapus URL
    text = re.sub(r'http\S+', '', text)
    
    # Menghapus tanda baca, tags, emoji, dan angka
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'&#[0-9]+;', '', text)
    text = re.sub(r'[0-9]+', '', text)
    
    return text

    # Fungsi untuk preprocessing stemming
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

# Fungsi untuk mengekstrak fitur dari teks komentar
def extract_features(words):
    return dict([(word, True) for word in words])

# Mengambil dataset twitter_samples
from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Menggabungkan dataset positif dan negatif
dataset = [(preprocess(tweet), 'Sentimen Positif') for tweet in positive_tweets] + [(preprocess(tweet), 'Sentimen Negatif') for tweet in negative_tweets]

# Mengacak urutan dataset
import random
random.shuffle(dataset)

# Mengambil daftar berisi semua kata dari seluruh tweet
all_words = [word.lower() for tweet, _ in dataset for word in word_tokenize(tweet)]
all_words = stem_tokens(all_words)  # Menggunakan stemming

# Mengambil 2000 kata unik yang paling umum
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:2000]

# Menghapus stop words dari kata-kata unik
stop_words = set(stopwords.words('english'))
word_features = [word for word in word_features if word not in stop_words]

# Menyiapkan data latih dan data uji
featuresets = [(extract_features(word_tokenize(tweet.lower())), sentiment) for (tweet, sentiment) in dataset]
train_set = featuresets[:int(len(featuresets) * 0.8)]
test_set = featuresets[int(len(featuresets) * 0.8):]

# Melatih klasifikasi Naive Bayes
classifier = NaiveBayesClassifier.train(train_set)

# Menghitung akurasi klasifikasi pada data uji
accuracy = nltk.classify.accuracy(classifier, test_set)

# Antarmuka Pengguna dengan Streamlit
st.title("Sentiment Analysis on Text")
st.write("Source of Dataset: NLTK twitter_samples")
st.write("Classification Accuracy on Test Data: {:.2%}".format(accuracy))

# Jendela cara penggunaan
with st.expander("How To Use"):
    st.write("1. Enter Text or Comments in the Input Field.")
    st.write("2. Click 'Analysis' Button to View the Analysis results.")
    st.write("3. The Analysis Results Will be Displayed Below")

# Input teks komentar dari pengguna
user_input = st.text_input("Enter the comment text:")

# Preprocessing teks
if st.button("Analysis "):
    if user_input:
        # Preprocessing teks
        preprocessed_steps = []

        
        # Langkah 1: Cleaning (menghapus tanda baca, angka, emoji)
        cleaned_input = re.sub(r'https?:\/\/\S+', '', user_input)  # Menghapus URL
        cleaned_input = re.sub(r'[' + string.punctuation + ']', ' ', cleaned_input)  # Menghapus tanda baca
        cleaned_input = re.sub(r'\w*\d\w*', '', cleaned_input)  # Menghapus angka

        preprocessed_steps.append("1. Cleaning (Removing URLs, Punctuation, Numbers): " + cleaned_input)

        # Langkah 2: Case Folding
        case_folded_input = cleaned_input.lower()

        preprocessed_steps.append("2. Case Folding: " + case_folded_input)


        # Langkah 3: Tokenisasi
        sample_words = word_tokenize(case_folded_input)
        tokenized_text = ", ".join(sample_words)  # Menggabungkan kata-kata dengan koma

        preprocessed_steps.append("3. Tokenisasi Text: " + tokenized_text)

        # Langkah 4: Menghapus stop words
        sample_words_no_stopwords = [word for word in sample_words if word not in stop_words]
        no_stopwords_text = ", ".join(sample_words_no_stopwords)
        
        preprocessed_steps.append("4. Remove Stop Words: " + no_stopwords_text)

        # Langkah 5: Stemming
        stemmer = PorterStemmer()
        stemmed_text = ", ".join([stemmer.stem(word) for word in sample_words_no_stopwords])
        
        preprocessed_steps.append("5. Stemming Text: " + stemmed_text)

        # Menampilkan langkah-langkah preprocessing
        st.write("Preprocessing steps:")
        for step in preprocessed_steps:
            st.write(step)

        # Melakukan analisis sentimen pada teks komentar media sosial
        sentiment = classifier.classify(extract_features(sample_words_no_stopwords))
        
        # Penjelasan mengapa teks terklasifikasikan sebagai sentimen positif atau negatif
        if sentiment == 'Sentiment Positive':
            st.write("Text classified as POSITIVE sentiment.")
            st.write("This may be due to the use of positive words, positive expressions, or a good context.")
        else:
            st.write("Text classified as NEGATIVE sentiment.")
            st.write("This may be due to the use of negative words, negative expressions, or poor context..")
        
        # Menampilkan kata-kata yang menjadi alasan terdeteksi sentimen negatif atau positif
        word_features_set = set(word_features)
        words_in_input = [word for word in sample_words if word in word_features_set]
        st.write("Words that contribute to sentiment analysis:", words_in_input)

        # Menampilkan kalimat dengan kata-kata penyebab
        st.write("related words:")
        words_highlighted = []
        for word in sample_words:
            if word in words_in_input:
                words_highlighted.append(f"<span style='background-color: #ffff00'>{word}</span>")
            else:
                words_highlighted.append(word)
        highlighted_sentence = " ".join(words_highlighted)
        st.markdown(highlighted_sentence, unsafe_allow_html=True)
