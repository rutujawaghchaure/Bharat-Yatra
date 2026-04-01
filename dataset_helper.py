import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px


def load_and_clean():
    df = pd.read_csv('final_travel_data_1000.csv')
    df.columns = df.columns.str.strip()
    df['Type'] = df['Type'].fillna('general').str.lower()
    df['Best Visit Time'] = df['Best Visit Time'].fillna('year-round').str.lower()
    df['State'] = df['State'].fillna('India').str.upper()
    
    def clean_val(v):
        n = re.findall(r'\d+', str(v).replace(',', ''))
        return (int(n[0]) + int(n[1]))/2 if len(n) >= 2 else (int(n[0]) if n else 0)

    df['Budget_Num'] = df['Trip Cost'].apply(clean_val)
    df['Days_Num'] = df['Stay Duration'].apply(clean_val)
    return df
# tfidf 
def get_recommendations(month, state, budget, days, interests):
    df = load_and_clean()
    
    # Filtering
    mask = (df['Best Visit Time'].str.contains(month.lower()) | (df['Best Visit Time'] == 'year-round')) & \
           (df['Budget_Num'] <= float(budget)) & \
           (df['Days_Num'] <= float(days))
    
    if state != "All India":
        mask = mask & (df['State'] == state)
        
    filtered = df[mask].copy()
    if filtered.empty: return []

    # ML Logic
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(filtered['Type'].str.replace('/', ' '))
    user_vec = tfidf.transform([" ".join(interests)])
    
    filtered['Score'] = cosine_similarity(user_vec, tfidf_matrix).flatten()
    return filtered.sort_values(by='Score', ascending=False).head(10).to_dict('records')

import plotly.express as px

# # random forest model
# def get_rf_recommendations(month, state, budget, days, interests):
#     df=load_and_clean()
#     mask = (df['Best Visit Time'].str.contains(month.lower()) | (df['Best Visit Time'] == 'year-round')) & \
#            (df['Budget_Num'] <= float(budget)) & \
#            (df['Days_Num'] <= float(days))
    
#     if state != "All India":
#         mask =mask & (df['State'] == state)
#     filtered = df[mask].copy()
#     if filtered.empty: return []    




def generate_budget_graph(df_filtered):
    if df_filtered.empty:
        return None
    
   
    fig = px.bar(df_filtered.head(10), 
                 x='Place Name', 
                 y='Budget_Num', 
                 color='State',
                 title="Budget Comparison of Recommended Places",
                 labels={'Budget_Num':'Estimated Cost (₹)', 'Place Name':'Destination'})
    
    
    graph_html = fig.to_html(full_html=False)
    return graph_html