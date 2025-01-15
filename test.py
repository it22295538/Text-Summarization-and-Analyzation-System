import spacy
import yake
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS

# Initialize Spacy for preprocessing
nlp = spacy.load('en_core_web_sm')

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize the summarization model from Hugging Face
summarizer = pipeline('summarization', model="t5-large")

# Helper function: Preprocess text with Spacy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Updated function to handle large text
def split_large_text(text, max_tokens=512):
    """
    Splits the input large text into manageable chunks for processing by models.
    Each chunk has a maximum of 1024 tokens.
    """
    sentences = [sent.text for sent in nlp(text).sents]  # Split into sentences

    current_chunk = []
    current_length = 0
    chunks = []

    for sentence in sentences:
        tokenized_sentence = summarizer.tokenizer.encode(sentence, return_tensors='pt')
        sentence_length = tokenized_sentence.shape[1]  # Number of tokens in the sentence

        if current_length + sentence_length > max_tokens:
            # Add the current chunk when the token limit is reached
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        # Add any remaining chunk
        chunks.append(' '.join(current_chunk))

    return chunks

# Function to ensure summary ends with a complete sentence
def ensure_complete_summary(summary):
    """
    Ensure that the summary does not end with an incomplete sentence.
    """
    doc = nlp(summary)
    sentences = list(doc.sents)

    # Check if the last sentence ends with proper punctuation
    if not sentences[-1].text.strip().endswith(('.', '!', '?')):
        # Remove the last sentence if it's incomplete
        summary = ' '.join([sent.text for sent in sentences[:-1]])

    # Check for grammatical completeness (basic approach)
    if summary and not summary.strip().endswith(('.', '!', '?')):
        summary += '.'

    return summary

# Function to summarize large texts with strict length control
def summarize_large_text(text, summary_type="medium", custom_length=None):
    """
    Summarize large input text by splitting it into chunks and summarizing each chunk.
    The length of the summary is controlled using fixed min/max word limits or custom length.
    """
    # Define length ranges for each summary type
    length_ranges = {
        "short": (40, 100),    # Short summary will have 40 to 100 words
        "medium": (100, 250),  # Medium summary will have 100 to 250 words
        "long": (250, 500)     # Long summary will have 250 to 500 words
    }

    if custom_length:
        min_length, max_length = custom_length
    else:
        min_length, max_length = length_ranges.get(summary_type, (250, 500))

    chunks = split_large_text(text, max_tokens=512)  # Split text into manageable chunks
    summaries = []

    for chunk in chunks:
        # Get the input length for the chunk
        input_length = len(chunk.split())

        # Adjust max_length if input_length is less than the set max_length
        adjusted_max_length = min(max_length, input_length)

        # Summarize each chunk with controlled max_length and min_length
        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    # Combine the summarized chunks
    combined_summary = ' '.join(summaries)

    # Ensure the final summary length is controlled based on words
    final_summary = ' '.join(combined_summary.split()[:max_length])  # Trim to the desired word count
    return remove_redundancy(final_summary)

# Function to remove redundant sentences
def remove_redundancy(summary):
    """
    Remove redundant sentences from the summary.
    """
    doc = nlp(summary)
    unique_sentences = []
    seen_sentences = set()

    for sent in doc.sents:
        clean_sentence = sent.text.strip()
        if clean_sentence.lower() not in seen_sentences:
            unique_sentences.append(clean_sentence)
            seen_sentences.add(clean_sentence.lower())

    return ' '.join(unique_sentences)

# Sentiment analysis for large text
def analyze_large_sentiment(text):
    """
    Perform sentiment analysis on large text.
    """
    return analyze_sentiment(text)

# Helper function: Sentiment Analysis using VADER
def analyze_sentiment(text):
    """
    Perform sentiment analysis using VADER.
    """
    scores = sentiment_analyzer.polarity_scores(text)
    compound_score = scores['compound']

    # Return classification based on compound score
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Keyword extraction for large text
def extract_keywords_from_large_text(text, max_keywords=10):
    """
    Extract keywords from large text using YAKE.
    """
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return [keyword for keyword, score in keywords[:max_keywords]]

# Topic modeling for large corpora
def topic_modeling_on_large_texts(texts, num_topics=3, num_words=5):
    """
    Perform topic modeling on large texts using LDA from Gensim.
    """
    # Preprocess all texts
    processed_texts = [preprocess_text(text) for text in texts]

    # Create a dictionary and bag-of-words corpus
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)

    # Get topics
    topics = lda_model.print_topics(num_words=num_words)

    # Clean and return topics
    cleaned_topics = []
    for topic in topics:
        topic_id, words_with_scores = topic
        # Extract words by splitting and cleaning the format
        words = [
            word_score.split("*")[1].strip().replace('"', '')  # Extract the word part
            for word_score in words_with_scores.split(" + ")  # Split words and scores
        ]

        # Generate a meaningful topic sentence
        topic_sentence = f"Topic {topic_id + 1}: {' '.join(words)}"

        # Append to the cleaned topics list
        cleaned_topics.append(topic_sentence)

    return cleaned_topics
