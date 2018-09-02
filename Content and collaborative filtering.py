
# coding: utf-8

# In[2]:


import os


# In[9]:


os.getcwd()


# In[10]:


os.chdir(r'C:\Users\suyas\Downloads\ml-1m\ml-1m')


# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[100]:



# Reading ratings file
# Ignore the timestamp column
ratings = pd.read_csv('RATINGS_CSV_FILE.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'rating'])

# Reading users file
users = pd.read_csv('USERS_CSV_FILE.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading movies file
movies = pd.read_csv('MOVIES_CSV_FILE.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])


# In[17]:


ratings.head()


# In[21]:


users.head()


# In[23]:


movies.head()


# In[26]:


# Import new libraries
get_ipython().magic('matplotlib inline')


# In[30]:


from wordcloud import WordCloud, STOPWORDS


# In[33]:


# Create a wordcloud of the movie titles
movies['title'] = movies['title'].fillna("").astype('str')


# In[35]:


title_corpus = ' '.join(movies['title'])


# In[37]:


title_wordcloud = WordCloud(stopwords=STOPWORDS,background_color='black',height=2000,width=4000).generate(title_corpus)


# In[50]:


# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# In[52]:


ratings['rating'].describe()


# In[54]:


# Import seaborn library
import seaborn as sns
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
get_ipython().magic('matplotlib inline')

# Display distribution of rating
sns.distplot(ratings['rating'].fillna(ratings['rating'].median()))


# In[56]:


# Join all 3 files into one dataframe
dataset = pd.merge(pd.merge(movies,ratings),users)


# In[59]:


# Display 20 movies with highest ratings
dataset[['title','genres','rating']].sort_values('rating',ascending=False).head(20)


# In[61]:


# Make a census of the genre keywords
genre_labels = set()
for s in movies['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))


# In[64]:


genre_labels


# In[66]:


# Function that counts the number of times each of the genre keywords appear
def count_word(dataset, ref_col, census):
    keyword_count = dict()
    for s in census: 
        keyword_count[s] = 0
    for census_keywords in dataset[ref_col].str.split('|'):        
        if type(census_keywords) == float and pd.isnull(census_keywords): 
            continue        
        for s in [s for s in census_keywords if s in census]: 
            if pd.notnull(s): 
                keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

# Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
keyword_occurences[:5]


# In[72]:


# Define the dictionary used to produce the genre wordcloud
genres = dict()
trunc_occurences = keyword_occurences[0:18]
for s in trunc_occurences:
    genres[s[0]] = s[1]


# In[75]:





# In[70]:


# Create the wordcloud
genre_wordcloud = WordCloud(width=1000,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres)


# In[76]:


# Plot the wordcloud
f, ax = plt.subplots(figsize=(16, 8))
plt.imshow(genre_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[104]:


# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')


# In[105]:


movies['genres']


# In[106]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape


# In[108]:


from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]


# In[110]:


# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])


# In[113]:


def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]


# In[123]:


genre_recommendations('Toy Story (1995)').head(20)


# In[125]:


Overall, here are the pros of using content-based recommendation:

No need for data on other users, thus no cold-start or sparsity problems.
Can recommend to users with unique tastes.
Can recommend new & unpopular items.
Can provide explanations for recommended items by listing content-features that caused an item to be recommended (in this case, movie genres)
However, there are some cons of using this approach:

Finding the appropriate features is hard.
Does not recommend items outside a user's content profile.
Unable to exploit quality judgments of other users.


# In[128]:


# Fill NaN values in user_id and movie_id column with 0
ratings['user_id'] = ratings['user_id'].fillna(0)
ratings['movie_id'] = ratings['movie_id'].fillna(0)


# In[130]:


# Replace NaN values in rating column with average of all values
ratings['rating'] = ratings['rating'].fillna(ratings['rating'].mean())


# In[132]:


# Randomly sample 1% of the ratings dataset
small_data = ratings.sample(frac=0.02)
# Check the sample info
print(small_data.info())


# In[134]:


from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(small_data, test_size=0.2)


# In[136]:


# Create two user-item matrices, one for training and another for testing
train_data_matrix = train_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])
test_data_matrix = test_data.as_matrix(columns = ['user_id', 'movie_id', 'rating'])

# Check their shape
print(train_data_matrix.shape)
print(test_data_matrix.shape)


# In[140]:



from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:4, :4])


# In[142]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation[:4, :4])


# In[144]:


# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[147]:


from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


# In[149]:


# Predict ratings on the training data with both similarity score
user_prediction = predict(train_data_matrix, user_correlation, type='user')
item_prediction = predict(train_data_matrix, item_correlation, type='item')

# RMSE on the test data
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# In[150]:


# RMSE on the train data
print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))

