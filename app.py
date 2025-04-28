#importing necessary libraries
import streamlit as st
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# Topic Modeling (using Latent Dirichlet Allocation - LDA)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
font_path = "https://github.com/Nikhil-Jagtap619/Market-Trend-Analysis-Prototype/blob/main/font/Oswald-VariableFont_wght.ttf"



# data cleaning, preprocessing the unstructure data into structure
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # Join the tokens back into a string
    text = " ".join(tokens)
    return text

#LDA
def get_topics(text):
   # Create a document-term matrix 
   vectorizer = CountVectorizer(stop_words='english')
   dtm = vectorizer.fit_transform(text)  # Use the preprocessed text
   
   LDA = LatentDirichletAllocation(n_components=5, random_state=42)  # Adjust n_components as needed
   LDA.fit(dtm)
   top_words = {}
   # Get the top words for each topic
   for index,topic in enumerate(LDA.components_):
    # print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    # print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    # print('\n')
    top_words[index] = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-15:]]
    # df = pd.DataFrame(top_words)
    return top_words



# get sentiments system # Returns a value between -1 (negative) and 1 (positive)
def get_sentiment(text):
  analysis = TextBlob(text)
  return analysis.sentiment.polarity  

def get_wordCloud(text):
   # Create a WordCloud object
   wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS, font_path=font_path).generate(text)
   return wordcloud



def main():
    st.title("Market Trend Analysis")
    st.write("Identifies the top 15 words for each discovered topic, providing insights into the prevalent themes.")
    user = st.text_area("Paste your text/para here for topic modeling")
    if st.button("Model the Topic"):
        temp_text = preprocess_text(user)
        topics = get_topics([temp_text])
        st.write(topics)
        # df = pd.date_range(topics, columns=["topics"])
        # st.table(df)

    st.title("Sentiments on the Paragraphs")
    st.subheader("The polarity score for text segment")
    st.write("Ranging from -1 to 1, where -1 represents negative sentiment and 1 represents positive sentiment")
    # user2 = st.text_area("Paste your chats/conversation here")
    if st.button("Get Sentiments"):
      result = round(get_sentiment(user),2)
      if result > 0 and result < 0.5:
        st.write(f"Weak Positive Sentiment: {result}")
      elif result > 0.5:
         st.write("Strong Positive Sentiments")
         st.success(result)
      elif result < 0 and result > -0.5:
        st.write(f"Weak Negative Sentimetns: {result}")
      elif result <= -1:
         st.write("Strong Negative Sentiments")
         st.error(result)
      else:
         st.write("Sentiments are neutral")

    st.title("Word Cloud")
    if st.button("Create WordCloud"):
       word_cloud = get_wordCloud(preprocess_text(user))
       fig, ax = plt.subplots(figsize=(10, 5))
       ax.imshow(word_cloud, interpolation="bilinear")
       ax.axis("off")
       st.pyplot(fig)
    else:
       st.write("Enter some text to generate a word cloud.")





    st.write("## Thank you for Visiting \nPrototype by Nikhil J")
    st.markdown("<h1 style='text-align: right; color: #d7e3fc; font-size: small;'><a href='https://github.com/Nikhil-Jagtap619/Market-Trend-Analysis-Prototype'>Looking for Source Code?</a></h1>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
