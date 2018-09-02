
# coding: utf-8

# In[4]:



# Import packages# Import 
import os
import pandas as pd


# In[14]:



# Define file directories# Define 
MOVIELENS_DIR = 'dat'
USER_DATA_FILE = r'C:\Users\suyas\Downloads\ml-1m\ml-1m\users.dat'
MOVIE_DATA_FILE = r'C:\Users\suyas\Downloads\ml-1m\ml-1m\movies.dat'
RATING_DATA_FILE = r'C:\Users\suyas\Downloads\ml-1m\ml-1m\ratings.dat'


# In[15]:


# Specify User's Age and Occupation Column
AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }


# In[17]:


# Define csv files to be saved into
USERS_CSV_FILE = 'users.csv'
MOVIES_CSV_FILE = 'movies.csv'
RATINGS_CSV_FILE = 'ratings.csv'


# In[21]:



ratings = pd.read_csv(os.path.join(MOVIELENS_DIR, RATING_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'movie_id', 'rating', 'timestamp'])


# In[26]:


# Set max_userid to the maximum user_id in the ratings
max_userid = ratings['user_id'].drop_duplicates().max()
# Set max_movieid to the maximum movie_id in the ratings
max_movieid = ratings['movie_id'].drop_duplicates().max()

# Process ratings dataframe for Keras Deep Learning model
# Add user_emb_id column whose values == user_id - 1
ratings['user_emb_id'] = ratings['user_id'] - 1
# Add movie_emb_id column whose values == movie_id - 1
ratings['movie_emb_id'] = ratings['movie_id'] - 1

len(ratings), 'ratings loaded'


# In[33]:



# Save into ratings.csv# Save i 
ratings.to_csv(r'C:\Users\suyas\Downloads\ml-1m\ml-1m\RATINGS_CSV_FILE.csv', 
               sep='\t', 
               header=True, 
               encoding='latin-1', 
               columns=['user_id', 'movie_id', 'rating', 'timestamp', 'user_emb_id', 'movie_emb_id'])
'Saved to', RATINGS_CSV_FILE


# In[35]:


# Read the Users File
users = pd.read_csv(os.path.join(MOVIELENS_DIR, USER_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
users['age_desc'] = users['age'].apply(lambda x: AGES[x])
users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])


# In[37]:


# Save into users.csv
users.to_csv(r'C:\Users\suyas\Downloads\ml-1m\ml-1m\USERS_CSV_FILE.csv', 
             sep='\t', 
             header=True, 
             encoding='latin-1',
             columns=['user_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])


# In[39]:


# Read the Movies File
movies = pd.read_csv(os.path.join(MOVIELENS_DIR, MOVIE_DATA_FILE), 
                    sep='::', 
                    engine='python', 
                    encoding='latin-1',
                    names=['movie_id', 'title', 'genres'])


# In[40]:


# Save into movies.csv
movies.to_csv(r'C:\Users\suyas\Downloads\ml-1m\ml-1m\MOVIES_CSV_FILE.csv', 
              sep='\t', 
              header=True, 
              columns=['movie_id', 'title', 'genres'])

