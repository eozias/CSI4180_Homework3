import evaluate
from datasets import load_dataset, metric
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import nltk
import collections
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer

dataset = load_dataset("OxAISH-AL-LLM/wiki_toxic", trust_remote_code=True)
tokenizerOne = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
tokenizerTwo = AutoTokenizer.from_pretrained("distilbert/distilbert-base-multilingual-cased")


def tokenize_function_one(examples):
    return tokenizerOne(examples["comment_text"], padding="max_length", truncation=True)


def tokenize_function_two(examples):
    return tokenizerTwo(examples["comment_text"], padding="max_length", truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def preprocess_text(text):
    # Convert text to lower
    text = text.lower()

    # Remove Punctuation
    tokenizer = RegexpTokenizer(r'\w+\'?\w+|\w+')
    words = tokenizer.tokenize(text)
    words_without_punctuation = [''.join(c for c in word if c not in string.punctuation or c in ["'", "’"]) for
                                 word in words]
    text = ' '.join(words_without_punctuation)

    # Lemmatize text
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmaWords = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
    text = ' '.join(lemmaWords)

    # Remove stop words
    stop_words_lower = set(word.lower() for word in stopwords.words('english'))
    stop_words_upper = set(word.title() for word in stopwords.words('english'))
    stop_words = stop_words_lower.union(stop_words_upper)
    tokens = word_tokenize(text)
    tokensNoSWs = [tok for tok in tokens if tok not in stop_words]
    processedText = ' '.join(tokensNoSWs)

    # Remove numbers
    processedText = re.sub(r'\d+', '', processedText)

    # Remove unimportant and common words, and singular letters from the text
    processedText = processedText.replace("reuters", "")
    processedText = processedText.replace("ap", "")
    processedText = processedText.replace("'s ", "")
    processedText = processedText.replace("washingtonpost", "")
    processedText = processedText.replace(" say ", "")
    processedText = processedText.replace(" new ", "")
    processedText = processedText.replace(" lt ", "")
    processedText = processedText.replace(" gt ", "")
    processedText = re.sub(r'\b\w\b', '', processedText)

    return processedText


# START OF "MAIN" FUNCTION

# 2.2 Fine-tuning pre-trained models

# Model One: google-bert/bert-base-cased
print("Model One: google-bert/bert-base-cased...")
tokenized_datasets_one = dataset.map(tokenize_function_one, batched=True)
small_train_dataset_one = tokenized_datasets_one["balanced_train"].shuffle(seed=42).select(range(1000))
small_eval_dataset_one = tokenized_datasets_one["test"].shuffle(seed=42).select(range(1000))
model_one = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=2)
training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")
trainer_one = Trainer(
    model=model_one,
    args=training_args,
    train_dataset=small_train_dataset_one,
    eval_dataset=small_eval_dataset_one,
    compute_metrics=compute_metrics,
)
trainer_one.train()

# Next four lines of code were written by GitHub Copilot
# This code is meant to predict the labels for the test split and then print the accuracy
predictions = trainer_one.predict(small_eval_dataset_one)
class_predictions = np.argmax(predictions.predictions, axis=-1)
eval_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print(eval_metrics)
print("\n")

# Model Two: distilbert/distilbert-base-multilingual-cased
print("Model Two: distilbert/distilbert-base-multilingual-cased...")
tokenized_datasets_two = dataset.map(tokenize_function_two, batched=True)
small_train_dataset_two = tokenized_datasets_two["balanced_train"].shuffle(seed=42).select(range(1000))
small_eval_dataset_two = tokenized_datasets_two["test"].shuffle(seed=42).select(range(1000))
model_two = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-multilingual-cased",
                                                               num_labels=2)
trainer_two = Trainer(
    model=model_two,
    args=training_args,
    train_dataset=small_train_dataset_two,
    eval_dataset=small_eval_dataset_two,
    compute_metrics=compute_metrics,
)
trainer_two.train()

# Next four lines of code were written by GitHub Copilot
# This code is meant to predict the labels for the test split and then print the accuracy
predictions = trainer_two.predict(small_eval_dataset_two)
class_predictions = np.argmax(predictions.predictions, axis=-1)
eval_metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print(eval_metrics)
print("\n")

# 2.3 Zero-shot classification is on Google Collab

# 2.4 Baselines
texts = dataset['balanced_train']['comment_text'][0:10000]
labels = dataset['balanced_train']['label'][0:10000]
processed_sentences = [];

for text in texts:
    processed_text = preprocess_text(text)
    processed_sentences.append(processed_text)

# Convert to Bag of Words Format
vocab = set()
bow_model = []
for text in processed_sentences:
    word_counts = {}
    tokens = nltk.word_tokenize(text)
    vocab.update(tokens)
    for word in tokens:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
    bow_model.append(word_counts)

# Preprocess and convert the test data into Bag of Words format
test_texts = dataset['test']['comment_text']
test_labels = dataset['test']['label']
processed_test_sentences = [preprocess_text(text) for text in test_texts]

bow_test_model = []
for text in processed_test_sentences:
    word_counts = collections.defaultdict(int)
    tokens = nltk.word_tokenize(text)
    for word in tokens:
        word_counts[word] += 1
    bow_test_model.append(word_counts)

# Convert the BoW model to a matrix format
dict_vectorizer = DictVectorizer()
bow_model_matrix = dict_vectorizer.fit_transform(bow_model)
bow_test_model_matrix = dict_vectorizer.transform(bow_test_model)

# Train the classifier
clf = MultinomialNB()
clf.fit(bow_model_matrix, labels)

# Make predictions
predictions = clf.predict(bow_test_model_matrix)

# Compute the accuracy
correct_predictions = sum(predicted == actual for predicted, actual in zip(predictions, test_labels))
accuracy = correct_predictions / len(test_labels)
print("The accuracy of the Multinomial Naive Bayes classifier is: \n")
print(f'Accuracy: {accuracy}')
