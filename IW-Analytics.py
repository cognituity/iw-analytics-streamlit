import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly
import plotly.graph_objs as go
from helper_functions.plotly_wordcloud import plotly_wordcloud as pwc
import random
import pyarrow.parquet as pq

page_title = 'Infowars Analytics'
st.set_page_config(page_title=page_title, page_icon="📈",layout="wide",initial_sidebar_state='collapsed')
st.markdown(f"# {page_title}")
st.sidebar.header(page_title)

@st.cache_data
def load_data(file_name):
    df = pq.read_pandas(f'data/{file_name}').to_pandas()
    
    return df

    
#word_seg_df = pq.read_pandas('data/ep_segments.parquet').to_pandas()
#word_seg_df = load_data('ep_segments.parquet')

#curse_word_df = pq.read_pandas('data/curse_words.parquet').to_pandas()
curse_word_df = load_data('curse_words.parquet')

#df = pq.read_pandas('data/monthly_sentiment.parquet').to_pandas()
monthly_sentiment_df = load_data('monthly_sentiment.parquet')

#wordcloud_sent_perc_df = pq.read_pandas('data/wordcloud_sent_perc.parquet').to_pandas()


perc_df = pd.read_csv("data/sentiment_perc.csv", engine='python')
perc_df['ep_dates'] = pd.to_datetime(perc_df['episode_month'])
min_date = perc_df['ep_dates'].dt.date.min()
max_date = perc_df['ep_dates'].dt.date.max()

sent_perc_df = perc_df.groupby('seg_start_perc')[['average_neg_sentiment','average_neu_sentiment','average_pos_sentiment','avg_count_store_mentions']].mean().reset_index()

st.markdown(f'''
    This is a text analysis of the Alex Jones Show on InfoWars. Episodes between {min_date} & {max_date}.

    *Note: This analysis is not restricted to words spoken by Alex Jones, only words spoken on the Alex Jones show (including advertisements by third party companies)*
    '''
)

with st.expander('About the page.'):
    st.markdown(f'''
    This page is in no way an endorsement of Infowars or Alex Jones. The show is a constant stream of misinformation that should probably not be consumed by anyone.

    I developed this website as part of a personal interest project to learn more about using speech to text engines & text analytics.

    The project has been primarily developed in Python & uses a wide variety of tools:
    - [OpenAI Whisper](https://github.com/openai/whisper) for speech to text
    - [Twitter-roBERTa-base](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) for sentiment analysis
    - [Postgres](https://www.postgresql.org)

    ''')
'''
with st.sidebar:
    with st.form(key='filter'):
        min_date_picker = st.date_input("Select a minimum date", value=min_date)
        max_date_picker = st.date_input("Select a max date", value=max_date)
        submit_button = st.form_submit_button(label='Apply')

'''
st.markdown(f'''### Average Sentiment Over Time

This chart shows the average sentiment of episodes over time.  
''')

df_unpivot = monthly_sentiment_df.groupby('episode_month')[['average_neg_sentiment', 'average_neu_sentiment', 'average_pos_sentiment']].mean().reset_index()
df_unpivot = pd.melt(df_unpivot, id_vars='episode_month', value_vars=['average_neg_sentiment', 'average_neu_sentiment', 'average_pos_sentiment'])
fig = px.line(df_unpivot, x="episode_month",y="value", color='variable', color_discrete_map={'avg_neg_sentiment':'red',
                                 'avg_neu_sentiment':'grey',
                                 'avg_pos_sentiment':'green'
                                 })
fig.update_layout(
    xaxis_title="Episode Month",
    yaxis_title="Average Sentiment",
    legend=dict(
    yanchor="bottom",
    xanchor="center"
))
st.plotly_chart(fig, theme="streamlit", use_container_width=True)


st.markdown('''### Average Sentiment & Store Mentions During an Episode

This chart shows the average sentiment journey of an episode along with how often the infowars store is mentioned.  
''')
m_sent_perc_df = pd.melt(sent_perc_df, id_vars='seg_start_perc', value_vars=['average_neg_sentiment', 'average_neu_sentiment', 'average_pos_sentiment','avg_count_store_mentions'])
perc_fig = px.line(m_sent_perc_df, x="seg_start_perc",y="value", color='variable', color_discrete_map={'average_neg_sentiment':'red',
                                'average_neu_sentiment':'grey',
                                'average_pos_sentiment':'green',
                                'avg_count_store_mentions':'blue'
                                })
perc_fig.update_layout(
    xaxis_title="Episode % Duration",
    yaxis_title="Average Sentiment",
    legend=dict(
        yanchor="bottom",
        xanchor="center"
        ))
st.plotly_chart(perc_fig, theme="streamlit", use_container_width=True)

st.markdown(f'''### Sentiment Wordclouds Throughout Duration

The below charts show clouds of the most negative, neutral & positive words at each percentage duration of a show.
''')

@st.cache_data(experimental_allow_widgets=True)
def gen_word_clouds():
    perc_slider = st.slider('Select a percentage duration', 0, 100, 0)
    perc_slider = perc_slider/100
    wordcloud_sent_perc_df = load_data('wordcloud_sent_perc.parquet')
    wordcloud_sent_perc_df = wordcloud_sent_perc_df[wordcloud_sent_perc_df['seg_start_perc'] == perc_slider]
    return wordcloud_sent_perc_df

wordcloud_sent_perc_df = gen_word_clouds()
neg_sent_perc_df = wordcloud_sent_perc_df[wordcloud_sent_perc_df['sent_type'] == 'neg']
neu_sent_perc_df = wordcloud_sent_perc_df[wordcloud_sent_perc_df['sent_type'] == 'neu']
pos_sent_perc_df = wordcloud_sent_perc_df[wordcloud_sent_perc_df['sent_type'] == 'pos']
wcol1, wcol2, wcol3 = st.columns(3)
with wcol1:
    st.markdown(f"#### Negative Sentiment Word Cloud")
    neg_word_list = neg_sent_perc_df['text_cleaned'].dropna().astype('string').values.tolist()
    neg_word_list = ' '.join(neg_word_list)
    neg_wrd_fig = pwc(neg_word_list)
    st.plotly_chart(neg_wrd_fig, theme="streamlit", use_container_width=True)
with wcol2:
    st.markdown(f"#### Neutral Sentiment Word Cloud")
    neu_word_list = neu_sent_perc_df['text_cleaned'].dropna().astype('string').values.tolist()
    neu_word_list = ' '.join(neu_word_list)
    neu_wrd_fig = pwc(neu_word_list)
    st.plotly_chart(neu_wrd_fig, theme="streamlit", use_container_width=True)
with wcol3:
    st.markdown(f"#### Positive Sentiment Word Cloud")
    pos_word_list = pos_sent_perc_df['text_cleaned'].dropna().astype('string').values.tolist()
    pos_word_list = ' '.join(pos_word_list)
    pos_wrd_fig = pwc(pos_word_list)
    st.plotly_chart(pos_wrd_fig, theme="streamlit", use_container_width=True)

st.markdown(f"### Sentiment Over Days Of Week")
day_df_unpivot = df.groupby(['episode_month','episode_day'])[['average_neg_sentiment', 'average_neu_sentiment', 'average_pos_sentiment']].mean().reset_index()
day_df_unpivot = pd.melt(day_df_unpivot, id_vars='episode_day', value_vars=['average_neg_sentiment', 'average_neu_sentiment', 'average_pos_sentiment'])
days_fig = px.box(day_df_unpivot,x='episode_day',y='value',color='variable')
days_fig.update_xaxes(categoryorder='array', categoryarray= ['sun','mon','tue','wed','thu','fri'])
st.plotly_chart(days_fig, theme="streamlit", use_container_width=True)

st.markdown(f"### Curse Counter")
with st.expander('Open to show NSFW Curse Counter.'):
    curse_df = curse_word_df.value_counts(subset=['episode_date','word']).reset_index()
    curse_df['episode_date'] = pd.to_datetime(curse_df['episode_date']).dt.to_period('M').dt.strftime("%Y-%m")
    curse_fig = px.bar(curse_df, x='episode_date',y=0, color='word')
    st.plotly_chart(curse_fig, theme="streamlit", use_container_width=True)
    st.markdown(f"*Based on George Carlin's Seven Dirty Words.*")
    