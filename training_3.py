import tensorflow as tf
import numpy as np
import json
import nltk
import transformers
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pickle

nltk.download('stopwords')


class ChatbotTrainer:
    def __init__(self, language):
        self.max_len = None
        self.language = language
        self.stemmer = SnowballStemmer(language) if language != "sw" else SnowballStemmer("english")
        self.words = []
        self.labels = []
        self.docs_x = []
        self.docs_y = []
        self.training = []
        self.output = []
        self.model = None

    def preprocess_intents(self, intents_file):
        with open(intents_file) as file:
            intents = json.load(file)

        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        for intent in intents["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                wrds = [lemmatizer.lemmatize(w.lower()) for w in wrds if w not in stop_words]
                self.words.extend(wrds)
                self.docs_x.append(wrds)
                self.docs_y.append(intent["tag"])
            if intent["tag"] not in self.labels:
                self.labels.append(intent["tag"])

        self.words = sorted(list(set(self.words)))
        self.labels = sorted(self.labels)
        out_empty = [0 for _ in range(len(self.labels))]
        for x, doc in enumerate(self.docs_x):
            bag = []
            for w in self.words:
                if w in doc:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row = out_empty[:]
            output_row[self.labels.index(self.docs_y[x])] = 1
            self.training.append(bag)
            self.output.append(output_row)

        self.training = np.array(self.training)
        self.output = np.array(self.output)

    def build_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(len(self.words) + 1, 100, input_length=len(self.training[0])))
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
        self.model.add(tf.keras.layers.Dense(len(self.labels), activation='softmax'))

    # def build_model(self):
    #     self.max_len = 20
    #     # Load pre-trained BERT model
    #     bert_model = transformers.TFBertModel.from_pretrained('bert-base-uncased')
    #     # Freeze the BERT model to reuse the pre-trained features
    #     bert_model.bert.trainable = False
    #
    #     input_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32)
    #     attention_mask = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32)
    #     token_type_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32)
    #
    #     sequence_output, pooled_output = bert_model(input_ids, attention_mask=attention_mask,
    #                                                 token_type_ids=token_type_ids)
    #
    #     # Add a trainable classification head on top of the BERT model
    #     x = tf.keras.layers.Dropout(0.3)(sequence_output)
    #     x = tf.keras.layers.GlobalAveragePooling1D()(x)
    #     x = tf.keras.layers.Dense(8, activation='relu')(x)
    #     x = tf.keras.layers.Dropout(0.3)(x)
    #     out = tf.keras.layers.Dense(len(self.labels), activation='softmax')(x)
    #
    #     self.model = tf.keras.models.Model(inputs=[input_ids, attention_mask, token_type_ids], outputs=out)

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs, batch_size):
        self.model.fit(self.training, self.output, epochs=epochs, batch_size=batch_size)

    def evaluate_model(self):
        return self.model.evaluate(self.training, self.output)

    def save_model(self, model_file, words_file, classes_file):
        self.model.save(model_file)
        pickle.dump(self.words, open(words_file, "wb"))
        pickle.dump(self.labels, open(classes_file, "wb"))

    def load_model(self, model_file, words_file, classes_file):
        self.model = tf.keras.models.load_model(model_file)
        self.words = pickle.load(open(words_file, "rb"))
        self.labels = pickle.load(open(classes_file, "rb"))

    def predict(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.stemmer.stem(word.lower()) for word in sentence_words]
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        bag = np.array(bag)
        res = self.model.predict(np.array([bag]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.labels[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, intents_file, sentence):
        global result
        results = self.predict(sentence)
        with open(intents_file) as file:
            intents = json.load(file)
        tag = results[0]['intent']
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = np.random.choice(i['responses'])
                break
        return result

    def chat(self, intents_file):
        print("Start talking with the bot (type quit to stop)!")
        while True:
            inp = input("You: ")
            if inp.lower() == "quit":
                break
            print("Bot: " + self.get_response(intents_file, inp))

    def train(self, intents_file, model_file, words_file, classes_file, epochs, batch_size):
        self.preprocess_intents(intents_file)
        self.build_model()
        self.compile_model()
        self.train_model(epochs, batch_size)
        self.save_model(model_file, words_file, classes_file)
        print("Model trained and saved successfully!")

    def load_and_chat(self, intents_file, model_file, words_file, classes_file):
        self.load_model(model_file, words_file, classes_file)
        self.chat(intents_file)


if __name__ == "__main__":
    language = "english"
    lang = "en"
    trainer = ChatbotTrainer(language)
    trainer.train(f"datasets/intents/customer_care.{lang}.json", f"models/chatbot/{lang}/2/eve_model.h5",
                  f"models/chatbot/{lang}/2/words.pkl", f"models/chatbot/{lang}/2/classes.pkl", 1100, 60)
    trainer.load_and_chat(
        f"datasets/intents/customer_care.{lang}.json",
        f"models/chatbot/{lang}/2/eve_model.h5",
        f"models/chatbot/{lang}/2/words.pkl",
        f"models/chatbot/{lang}/2/classes.pkl",
    )
