import pandas as pd 
import os 
import nltk
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('stopwords')
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

# A class that preprocesses text and calculates its sentiment score using the SentimentIntensityAnalyzer class from the NLTK library.

class SentimentalText:
    
    def __init__(self, language='english', translate=False):#Initializes the SentimentalText object.
        self.language = language
        self.translate = translate
        self.sia = SentimentIntensityAnalyzer()

    # Preprocesses a text by removing stop words, tokenizing it, and converting all words to lowercase
    # Returns a string with the preprocessed text.

    def pre_processamento(self, frase):  
        if self.translate:
            frase = GoogleTranslator(source='auto', target=self.language).translate(frase)

        stop_words = set(stopwords.words(self.language))
        word_tokens = word_tokenize(frase.lower())
        filtered_sentence = [word for word in word_tokens if word.isalnum() and word not in stop_words]
        return ' '.join(filtered_sentence)
    
    # Calculates the Sentiment Score using NLTK Library
        # Float -1 to 1, where -1 is a very negative comment and 1 is a positive

    def sentimento(self, frase=''):
        preprocessed_sentence = self.pre_processamento(frase)
        sentiment_score = self.sia.polarity_scores(preprocessed_sentence)
        return sentiment_score['compound']


if __name__ == '__main__':
    df = pd.read_csv('amazon.csv')
    df['text'] = df['reviewTitle'] + ' ' + df['reviewDescription']
    df['sentiment_score'] = df['text'].apply(lambda x: SentimentalText(language='english', translate=True).sentimento(x))

    # Categorizing scores

    def score_categories(score):
        if score > 0.5:
            return "Very positive comment"
        elif score > 0:
            return "Positive comment"
        elif score < -0.5:
            return "Very negative comment"
        else:
            return "Negative comment"
    
    df['sentiment_categories'] = df['sentiment_score'].apply(score_categories)
    df_output = df[['text', 'sentiment_score']]
    df.to_csv('review_score.csv', index=False)

    # Creating a folder to save the plots

    if not os.path.exists("plots"):
        os.makedirs("plots")

    # Creating a plot to the review scores(WIP: word plots are still kinda wrong)

    comments = df['text'].tolist()
    sentiment_scores = df['sentiment_score'].tolist()
    plt.figure(figsize=(20, 12))
    plt.bar(range(len(sentiment_scores)), sentiment_scores, color=['red' if s < 0 else 'green' for s in sentiment_scores])
    plt.xticks(range(len(comments)), comments, rotation=45, fontsize=6)
    plt.xlabel('Comment')
    plt.ylabel('Sentiment Score')
    plt.savefig(f"plots/scores.png", bbox_inches='tight')
    plt.show()

    # Using Counter to count the most used words inside the amazon.csv
    # Also removes words that are not necessary 

    stopwords_list = set(stopwords.words('english'))
    unnecessary_words = {'e', 'para', 'de', 'esse', 'eu', 'nos', 'um', 'onde', 'alguns', 'uma',
                        'é', 'vou', 'da', 'em', 'sem', 'já', 'dos', 'do', 'na', 'nas', '"de', 'que', 'estou'}

    # Logic for wordcloud for each comment
    for i, review in enumerate(df['text']):
        all_words = [word.lower() for word in review.split() if word.lower() not in stopwords_list and word.lower() not in unnecessary_words]
        word_counts = Counter(all_words)
        print(f"Palavras mais comuns no comentário {i}:")
        for word, count in word_counts.most_common(20):
            print(f"{word}: {count}")

        # Generate wordcloud for each comment
        word_cloud = WordCloud(width=800, height=400, background_color='gray').generate_from_frequencies(word_counts)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(f"plots/wordcloud_{i}.png", bbox_inches='tight')        
        plt.show()

    # Logic for all comments combined
    all_words = []
    for review in df['text']:
        all_words.extend([word.lower() for word in review.split() if word.lower() not in stopwords_list and word.lower() not in unnecessary_words])
    word_counts = Counter(all_words)
    print("Palavras mais comuns em todos os comentários:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")

    # Generate wordcloud for all comments combined
    word_cloud = WordCloud(width=800, height=400, background_color='gray').generate_from_frequencies(word_counts)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("plots/wordcloud_all.png", bbox_inches='tight')
    plt.show()

    print(df)