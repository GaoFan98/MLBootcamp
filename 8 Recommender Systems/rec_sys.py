import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colum_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('u.data', sep='\t', names=colum_names)

movie_titles = pd.read_csv('Movie_Id_Titles')

# MERGING TITLES OF MOVIES COLUMN VALUES WITH RELEVANT ITEM ID
df = pd.merge(df, movie_titles, on='item_id')

sns.set_style('white')

# CREATING RATINGS DATAFRAME WITH AVERAGE RATING AND NUMBER OF RATINGS
# AVERAGE TOP RATING MOVIES
av_rating = df.groupby('title')['rating'].mean().sort_values(ascending=False)
# COUNT OF RATING GIVEN TO MOVIE
most_rating = df.groupby('title')['rating'].count().sort_values(ascending=False)
# CONCAT OF RATINGS AND NUMBER OF RATINGS
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

# VISUALIZATION OF RATINGS AND NUMBER OF RATINGS
sns.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)

# CREATING MATRIX OF DATA
moviemat = df.pivot_table(index='user_id', columns='title', values='rating')

# GRAB 2 MOVIES DATA
starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
# CHECKING SIMILARITY WITH OTHER MOVIES
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
# CLEANING DATA FROM NULL VALUES
corr_starwars = pd.DataFrame(similar_to_starwars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
# FILTERING DATA AND GETTING CORRELATION TO STARWARS MOVIE
corr_starwars = corr_starwars.join(ratings['num of ratings'])
# FILTER OUT WITH MORE THAN 100 RATING NUMBER
most_similar_to_starwars = corr_starwars[corr_starwars['num of ratings'] > 100].sort_values('Correlation',
                                                                                            ascending=False)
# DO THE SAME THING WITH LIAR LIAR MOVIE
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
# CLEANING DATA FROM NULL VALUES
corr_liarliar = pd.DataFrame(similar_to_liarliar, columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
# FILTERING DATA AND GETTING CORRELATION TO STARWARS MOVIE
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])
# FILTER OUT WITH MORE THAN 100 RATING NUMBER
most_similar_to_liarliar = corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlation',
                                                                                            ascending=False)
print(most_similar_to_liarliar)
