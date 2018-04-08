

```python
# 2018-04-09 -- Steve Dow -- Homework
# ## Unit 7 | Assignment - Distinguishing Sentiments

# Dow - THREE TAKE AWAYS
# (1) Based on a review of several runs of this script, the New York Times tends to be the least negative (on Twitter) of the news agencies reviewed.
# (2) In a previous run of this script, FoxNews was the most negative by far. The latest script has them much less negative. This makes me think that there are several other factors that likely impact this analysis, including time of day... and specific story covered. 
# (3) Based on the script runs to date, CBS News was surprising more negative in sentiment than I anticipated.



# INSTRUCTIONS
# In this assignment, you'll create a Python script to perform a sentiment analysis of the Twitter activity of various 
# news oulets, and to present your findings visually.

# Your final output should provide a visualized summary of the sentiments expressed in Tweets 
# sent out by the following news organizations: __BBC, CBS, CNN, Fox, and New York times__.

# (Deliverable 1) The first plot will be and/or feature the following:
# * Be a scatter plot of sentiments of the last __100__ tweets sent out by each news organization, 
#   ranging from -1.0 to 1.0, where a score of 0 expresses a neutral sentiment, -1 the most negative sentiment possible, 
#   and +1 the most positive sentiment possible.
# * Each plot point will reflect the _compound_ sentiment of a tweet.
# * Sort each plot point by its relative timestamp.

# (Deliverable 2) The second plot will be a bar plot visualizing the _overall_ sentiments of the last 100 tweets from each organization. 
# For this plot, you will again aggregate the compound sentiments analyzed by VADER.

# The tools of the trade you will need for your task as a data analyst include the following: 
# tweepy, pandas, matplotlib, seaborn, textblob, and VADER.

# Your final Jupyter notebook must:
# * Pull last 100 tweets from each outlet.
# * Perform a sentiment analysis with the compound, positive, neutral, and negative scoring for each tweet.
# * Pull into a DataFrame the tweet's source acount, its text, its date, and its compound, positive, neutral, 
#   and negative sentiment scores.
# * Export the data in the DataFrame into a CSV file.
# * Save PNG images for each plot.

# As final considerations:
# * Use the Matplotlib and Seaborn libraries.
# (Deliverable 3) * Include a written description of three observable trends based on the data.
# * Include proper labeling of your plots, including plot titles (with date of analysis) and axes labels.
# (Deliverable 4) * Include an exported markdown version of your Notebook called  `README.md` in your GitHub repository.

```


```python
# Dependencies
import pandas as pd
import tweepy
import time
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime


# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
consumer_key = "1FDMEahjFGKkgQbhA4aBoChbl"
consumer_secret = "ifGWFwIaLddfsOAyFpxg1VXoi7Y3O4lUpMe0ss1NPMHwwDv7qI"
access_token = "1412065645-qJM4pv2r5Xb84H2Fg2tTbzzWilCTTqSNjpQCAuz"
access_token_secret = "F8sXI3t0SOv5LcDY37zao3UpZnfUHps8aKOzpYphKicPy"

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

```


```python
# Target User Accounts
target_user = ("@BBCWorld","@CBSNews","@cnnbrk","@nytimes","@FoxNews")
#target_user = ("@CBSNews")

usernames = []
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
text_list = []
date_list = []
converted_date_list = []
time_diff_list = []


# Loop through each user
for user in target_user:

    #time_diff = 0 
   
    
    # Variables for holding sentiments

    # 20 tweets/page x 5 pages = 100 tweets
    for x in range(1,6):

        # Get all tweets from home feed
        public_tweets = api.user_timeline(user, page=x)

        # Loop through all tweets
        for tweet in public_tweets:

            
            # Run Vader Analysis on each tweet
            results = analyzer.polarity_scores(tweet["text"])
            compound = results["compound"]
            pos = results["pos"]
            neu = results["neu"]
            neg = results["neg"]
            
            text = tweet["text"]
            date = tweet["created_at"]
            converted_date = datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y")
            #time_diff = converted_date - time_diff

            # Add each value to the appropriate list
            usernames.append(user)
            compound_list.append(compound)
            positive_list.append(pos)
            negative_list.append(neg)
            neutral_list.append(neu)
            text_list.append(text)
            date_list.append(date)
            converted_date_list.append(converted_date)
            #time_diff_list.append(time_diff)
            
# Create a dictionary of results
dict_results = {
 "Username": usernames,
 "Compound Score": compound_list,
 "Postive Score": positive_list,
 "Neutral Score": neutral_list,
 "Negative Score": negative_list,
 "Text": text_list,
 "Date": converted_date_list,
# "Time Diff": time_diff_list
 }


```


```python
# Create DataFrame
results_df = pd.DataFrame(dict_results).set_index("Username").round(2)
results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound Score</th>
      <th>Date</th>
      <th>Negative Score</th>
      <th>Neutral Score</th>
      <th>Postive Score</th>
      <th>Text</th>
    </tr>
    <tr>
      <th>Username</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>@BBCWorld</th>
      <td>0.18</td>
      <td>2018-04-08 03:40:10+00:00</td>
      <td>0.27</td>
      <td>0.40</td>
      <td>0.34</td>
      <td>Egypt's problem with sexy cinema https://t.co/...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.48</td>
      <td>2018-04-08 03:04:01+00:00</td>
      <td>0.24</td>
      <td>0.76</td>
      <td>0.00</td>
      <td>Japanese father arrested for caging son for ov...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.62</td>
      <td>2018-04-08 01:18:36+00:00</td>
      <td>0.27</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>How the crisis in the Gulf could spread to Eas...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.60</td>
      <td>2018-04-08 01:15:33+00:00</td>
      <td>0.38</td>
      <td>0.62</td>
      <td>0.00</td>
      <td>Obituary: Keith Murdoch, the disgraced All Bla...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.36</td>
      <td>2018-04-08 01:15:31+00:00</td>
      <td>0.26</td>
      <td>0.74</td>
      <td>0.00</td>
      <td>How 'condom snorting' turned into a pro-gun ar...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-08 01:09:24+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Inside the White House Bible Study group https...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.51</td>
      <td>2018-04-08 01:04:19+00:00</td>
      <td>0.00</td>
      <td>0.73</td>
      <td>0.27</td>
      <td>'I paid $90,000 to free my family from IS' htt...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.91</td>
      <td>2018-04-08 00:33:12+00:00</td>
      <td>0.53</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>Syria war: Scores dead in Syria gas attack, re...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 22:46:40+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Trump Tower fires: Blaze breaks out at New Yor...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 22:04:19+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Brazil's Lula surrenders to police https://t.c...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.87</td>
      <td>2018-04-07 20:41:19+00:00</td>
      <td>0.55</td>
      <td>0.45</td>
      <td>0.00</td>
      <td>Canada hockey team crash: 'Entire country in s...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.34</td>
      <td>2018-04-07 20:18:03+00:00</td>
      <td>0.26</td>
      <td>0.74</td>
      <td>0.00</td>
      <td>Vatican police arrest ex-diplomat over 'child ...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.66</td>
      <td>2018-04-07 16:53:36+00:00</td>
      <td>0.39</td>
      <td>0.61</td>
      <td>0.00</td>
      <td>Israel to investigate killing of Palestinian j...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 15:57:19+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Lula to surrender to police https://t.co/ksRE4...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 15:10:36+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Prince of Wales gets new chief title in ceremo...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.76</td>
      <td>2018-04-07 15:07:25+00:00</td>
      <td>0.31</td>
      <td>0.69</td>
      <td>0.00</td>
      <td>RT @BBCBreaking: German Police say they believ...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 14:50:14+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Van drives into pedestrians in Germany https:/...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 14:46:08+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>RT @BBCBreaking: Reports of casualties in Germ...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 14:41:20+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>The migrant caravan from Central America trave...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.73</td>
      <td>2018-04-07 13:58:36+00:00</td>
      <td>0.24</td>
      <td>0.71</td>
      <td>0.05</td>
      <td>RT @BBCSport: A young French footballer has di...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>-0.05</td>
      <td>2018-04-07 12:12:02+00:00</td>
      <td>0.09</td>
      <td>0.91</td>
      <td>0.00</td>
      <td>Facebook suspends AIQ data firm used by Vote L...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 12:06:31+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Prince Charles gets new Chief title on Vanuatu...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.10</td>
      <td>2018-04-07 12:00:24+00:00</td>
      <td>0.00</td>
      <td>0.85</td>
      <td>0.15</td>
      <td>Samba Diop: Le Havre defender dies aged 18 htt...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 12:00:24+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Grass skirt in Vanuatu for 'high chief' Prince...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 11:45:19+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Mexico's president @EPN quotes JFK during nati...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 11:17:16+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Lula: Brazil ex-president's police stand-off h...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 10:39:16+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Mexico to Trump: Don't take your 'frustrations...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.64</td>
      <td>2018-04-07 10:30:23+00:00</td>
      <td>0.00</td>
      <td>0.78</td>
      <td>0.22</td>
      <td>RT @BBCIndia: Huge crowds who braved scorching...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 10:13:02+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Salman Khan, Bollywood superstar, bailed in po...</td>
    </tr>
    <tr>
      <th>@BBCWorld</th>
      <td>0.00</td>
      <td>2018-04-07 08:51:03+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Russell Crowe 'divorce auction': Memorabilia u...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.49</td>
      <td>2018-04-07 23:03:15+00:00</td>
      <td>0.00</td>
      <td>0.79</td>
      <td>0.21</td>
      <td>Seven-year-old girl hikes Mount Kilimanjaro in...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.13</td>
      <td>2018-04-07 22:51:34+00:00</td>
      <td>0.21</td>
      <td>0.60</td>
      <td>0.19</td>
      <td>Moments ago, President @realDonaldTrump thanke...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.34</td>
      <td>2018-04-07 22:45:36+00:00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>The @NYPDnews and @FDNY are on the scene of a ...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.54</td>
      <td>2018-04-07 22:41:10+00:00</td>
      <td>0.21</td>
      <td>0.79</td>
      <td>0.00</td>
      <td>BREAKING: The @FDNY is battling a multi-alarm ...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.92</td>
      <td>2018-04-07 22:33:33+00:00</td>
      <td>0.52</td>
      <td>0.48</td>
      <td>0.00</td>
      <td>Victims slowly identified in Canadian hockey t...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.34</td>
      <td>2018-04-07 22:26:04+00:00</td>
      <td>0.21</td>
      <td>0.79</td>
      <td>0.00</td>
      <td>Fire erupts at Trump Tower in New York City ht...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.15</td>
      <td>2018-04-07 22:23:26+00:00</td>
      <td>0.08</td>
      <td>0.92</td>
      <td>0.00</td>
      <td>.@susanferrechio: "If Pruitt goes, whoever com...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.30</td>
      <td>2018-04-07 22:16:25+00:00</td>
      <td>0.16</td>
      <td>0.84</td>
      <td>0.00</td>
      <td>.@realDonaldTrump slams DOJ, FBI over missed d...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.73</td>
      <td>2018-04-07 22:09:18+00:00</td>
      <td>0.00</td>
      <td>0.64</td>
      <td>0.36</td>
      <td>.@RepKevinCramer expected to win GOP endorseme...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.76</td>
      <td>2018-04-07 22:01:26+00:00</td>
      <td>0.39</td>
      <td>0.61</td>
      <td>0.00</td>
      <td>.@POTUS zeroes in on tackling illegal immigrat...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.00</td>
      <td>2018-04-07 21:53:49+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Volcanic lightning is seen above Shinmoedake p...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.00</td>
      <td>2018-04-07 21:53:17+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>Earlier, @PressSec reiterated the Trump admini...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.72</td>
      <td>2018-04-07 21:50:00+00:00</td>
      <td>0.00</td>
      <td>0.68</td>
      <td>0.32</td>
      <td>.@Nigel_Farage: “In terms of foreign policy, t...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.48</td>
      <td>2018-04-07 21:45:00+00:00</td>
      <td>0.20</td>
      <td>0.80</td>
      <td>0.00</td>
      <td>.@Nigel_Farage: “I think with Barack Obama… Am...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.62</td>
      <td>2018-04-07 21:40:00+00:00</td>
      <td>0.00</td>
      <td>0.69</td>
      <td>0.31</td>
      <td>.@Reince: “I think Scott Pruitt is doing a gre...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.66</td>
      <td>2018-04-07 21:33:26+00:00</td>
      <td>0.00</td>
      <td>0.71</td>
      <td>0.29</td>
      <td>.@GovMikeHuckabee is in favor President @realD...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.00</td>
      <td>2018-04-07 21:31:00+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>.@DiamondandSilk: "Long gone are the days that...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.76</td>
      <td>2018-04-07 21:29:15+00:00</td>
      <td>0.27</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>BREAKING: Death toll in Canadian hockey team b...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.80</td>
      <td>2018-04-07 21:25:46+00:00</td>
      <td>0.38</td>
      <td>0.62</td>
      <td>0.00</td>
      <td>'Madness': Family outraged that 4 teens senten...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.59</td>
      <td>2018-04-07 21:25:36+00:00</td>
      <td>0.27</td>
      <td>0.62</td>
      <td>0.12</td>
      <td>Pilots warned of safety concerns before helico...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.63</td>
      <td>2018-04-07 21:17:57+00:00</td>
      <td>0.00</td>
      <td>0.80</td>
      <td>0.20</td>
      <td>“I really feel that God opened the door for ou...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.51</td>
      <td>2018-04-07 21:12:53+00:00</td>
      <td>0.15</td>
      <td>0.85</td>
      <td>0.00</td>
      <td>.@Nigel_Farage: "The liberal media, they're so...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.36</td>
      <td>2018-04-07 21:11:08+00:00</td>
      <td>0.13</td>
      <td>0.87</td>
      <td>0.00</td>
      <td>On "Cavuto Live," Escondido @MayorSamAbed told...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.25</td>
      <td>2018-04-07 21:06:09+00:00</td>
      <td>0.19</td>
      <td>0.68</td>
      <td>0.13</td>
      <td>California deputy arrested for alleged 'unlawf...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.90</td>
      <td>2018-04-07 20:56:29+00:00</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>Crash of hockey team's bus leaves at least 14 ...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.54</td>
      <td>2018-04-07 20:48:12+00:00</td>
      <td>0.17</td>
      <td>0.83</td>
      <td>0.00</td>
      <td>Qantas' former 'poster girl' flight attendant ...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.00</td>
      <td>2018-04-07 20:47:25+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>RT @FoxBusiness: CBS said to plan all-stock bi...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.49</td>
      <td>2018-04-07 20:47:21+00:00</td>
      <td>0.15</td>
      <td>0.85</td>
      <td>0.00</td>
      <td>RT @FoxBusiness: .@LouDobbs: "California state...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>0.00</td>
      <td>2018-04-07 20:47:18+00:00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>RT @FoxBusiness: High-tax states are already t...</td>
    </tr>
    <tr>
      <th>@FoxNews</th>
      <td>-0.53</td>
      <td>2018-04-07 20:45:37+00:00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>Jonathan Schanzer: "We need to take a look at ...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 6 columns</p>
</div>




```python
# Finally, export this file 
results_df.to_csv("2018-04-09-Dow-NewsStats.csv", index=False)

```


```python
# Create a scatter plot 
#results_df.plot(kind="scatter", x="Date", y="Compound Score", grid=True, figsize=(20,10),
#              title="Sentiment Analysis of Media Tweets (4/06/2018)")
#look at datetime stamping import date? - look at weathe rone solved
#plt.show()
```


```python
# Split up our data into groups
results_grouped_by_user = results_df.groupby("Username").mean()
grouped_sentiment = results_grouped_by_user["Compound Score"]

# Chart our data, give it a title, and label the axes
results_chart = grouped_sentiment.plot(kind="bar", title="Sentiment Analysis")
results_chart.set_xlabel("News Agency")
results_chart.set_ylabel("Overall Sentiment")
plt.savefig("2018-04-09-Dow-News-Sentiment.png")
plt.show()
```


![png](output_7_0.png)

