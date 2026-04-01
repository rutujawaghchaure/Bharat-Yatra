# from flask import Flask, render_template, request
# import pandas as pd
# import re
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from dataset_helper import get_recommendations, load_and_clean, generate_budget_graph


# app = Flask(__name__)

# # --- DATA LOADING & CLEANING ---
# def load_and_clean():
#     df = pd.read_csv('final_travel_data_1000.csv')
#     df.columns = df.columns.str.strip()
#     df['Type'] = df['Type'].fillna('general').str.lower()
#     df['Best Visit Time'] = df['Best Visit Time'].fillna('year-round').str.lower()
#     df['State'] = df['State'].fillna('India').str.upper()
    
#     # Cost cleaning logic
#     def clean_cost(val):
#         nums = re.findall(r'\d+', str(val).replace(',', ''))
#         if len(nums) >= 2:
#             return (int(nums[0]) + int(nums[1])) / 2
#         return int(nums[0]) if nums else 0

#     df['Budget_Num'] = df['Trip Cost'].apply(clean_cost)
#     return df

# df_master = load_and_clean()

# # --- RECOMMENDATION LOGIC ---
# def get_recommendations(month, state, budget, interests):
#     df = df_master.copy()
    
#     # Primary Filters (Month & Budget)
#     mask = (df['Best Visit Time'].str.contains(month.lower()) | (df['Best Visit Time'] == 'year-round')) & \
#            (df['Budget_Num'] <= float(budget))
    
#     # State Filter
#     if state != "All India":
#         mask = mask & (df['State'] == state)
        
#     filtered = df[mask].copy()
#     if filtered.empty: return []

#     # ML Logic (TF-IDF) for Interest Matching
#     tfidf = TfidfVectorizer(stop_words='english')
#     tfidf_matrix = tfidf.fit_transform(filtered['Type'].str.replace('/', ' '))
#     user_vec = tfidf.transform([interests.lower()])
    
#     filtered['Score'] = cosine_similarity(user_vec, tfidf_matrix).flatten()
#     # Top 12 results return karein
#     return filtered.sort_values(by='Score', ascending=False).head(12).to_dict('records')

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     states = sorted(df_master['State'].unique())
#     months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
#     results = None

#     if request.method == 'POST':
#         month = request.form.get('month')
#         state = request.form.get('state')
#         budget = request.form.get('budget', 5000)
#         interests = request.form.get('interests', 'nature')
        
#         results = get_recommendations(month, state, budget, interests)

#     return render_template('index.html', states=states, months=months, results=results)

# # from dataset_helper import get_recommendations, load_and_clean, generate_budget_graph

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     # ... purana code ...
#     graph = None
#     if request.method == 'POST':
#         # ... inputs lene wala code ...
#         results_data = get_recommendations(month, state, budget, interests)
#         results_df = pd.DataFrame(results_data)
        
#         # Graph generate karein
#         graph = generate_budget_graph(results_df)
#         results = results_data # Dictionary format for template

#     return render_template('index.html', states=states, months=months, results=results, graph=graph)
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataset_helper import generate_budget_graph

import pymysql # Database create karne ke liye zaroori
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# ==========================================
# 🛑 AUTO-CREATE DATABASE LOGIC (MySQL) 🛑
# ==========================================
# Ye function check karega ki database hai ya nahi. Agar nahi, toh bana dega.
def create_database_if_not_exists():
    try:
        # XAMPP default connection (Bina database naam ke connect karo)
        conn = pymysql.connect(host='localhost', user='root', password='')
        cursor = conn.cursor()
        # Database banane ki SQL query chalao
        cursor.execute("CREATE DATABASE IF NOT EXISTS bharat_yatra")
        conn.commit()
        cursor.close()
        conn.close()
        print("Database 'bharat_yatra' is ready!")
    except Exception as e:
        print(f"Error creating database: {e}")
        print("Kripya check karein ki XAMPP mein MySQL start hai ya nahi.")

# App start hone se pehle database banana zaroori hai
create_database_if_not_exists()

# ==========================================
# 🛑 FLASK-SQLALCHEMY SETUP 🛑
# ==========================================
app.config['SECRET_KEY'] = 'super_secret_key_123' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/bharat_yatra' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 

# --- DATABASE MODEL (User Table) ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- 1. DATA LOADING & CLEANING ---
def load_and_clean():
    try:
        df = pd.read_csv('final_travel_data_1000.csv')
        df.columns = df.columns.str.strip()
        df['Type'] = df['Type'].fillna('general').str.lower()
        df['Best Visit Time'] = df['Best Visit Time'].fillna('year-round').str.lower()
        df['State'] = df['State'].fillna('India').str.upper()
        
        def clean_cost(val):
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            if len(nums) >= 2:
                return (int(nums[0]) + int(nums[1])) / 2
            return int(nums[0]) if nums else 0

        df['Budget_Num'] = df['Trip Cost'].apply(clean_cost)
        return df
    except FileNotFoundError:
        print("Error: final_travel_data_1000.csv file not found!")
        return pd.DataFrame()

df_master = load_and_clean()

# --- 2. RECOMMENDATION LOGIC ---
def get_recommendations(month, state, budget, interests):
    if df_master.empty: return pd.DataFrame()
    
    df = df_master.copy()
    
    mask = (df['Best Visit Time'].str.contains(month.lower()) | (df['Best Visit Time'] == 'year-round')) & \
           (df['Budget_Num'] <= float(budget))
    
    if state != "All India":
        mask = mask & (df['State'] == state)
        
    filtered = df[mask].copy()
    if filtered.empty: 
        return pd.DataFrame() 

    tfidf = TfidfVectorizer(stop_words='english')
    text_data = filtered['Type'].str.replace('/', ' ')
    text_data = text_data.fillna('general') 
    
    tfidf_matrix = tfidf.fit_transform(text_data)
    user_vec = tfidf.transform([interests.lower()])
    
    filtered['Score'] = cosine_similarity(user_vec, tfidf_matrix).flatten()
    return filtered.sort_values(by='Score', ascending=False).head(12)


# ==========================================
# AUTHENTICATION ROUTES
# ==========================================
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists. Please login.', 'error')
            return redirect(url_for('signup'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(name=name, email=email, password=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
        
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            flash('Please check your login details and try again.', 'error')
            return redirect(url_for('login'))

        login_user(user)
        return redirect(url_for('index'))
        
    return render_template('login.html')

@app.route('/logout')
@login_required 
def logout():
    logout_user()
    return redirect(url_for('login'))


# --- 3. MAIN ROUTE ---
@app.route('/', methods=['GET', 'POST'])
@login_required 
def index():
    states = []
    if not df_master.empty and 'State' in df_master.columns:
        states = sorted(df_master['State'].unique())
        
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    results = None
    graph = None

    if request.method == 'POST':
        month = request.form.get('month')
        if not month: month = "year-round"
        
        state = request.form.get('state')
        if not state: state = "All India"
        
        budget_input = request.form.get('budget')
        budget = budget_input if budget_input and str(budget_input).strip() != "" else 5000
        
        interests = request.form.get('interests')
        if not interests: interests = "nature"
        
        results_df = get_recommendations(month, state, budget, interests)
        
        if not results_df.empty:
            results = results_df.to_dict('records')
            try:
                graph = generate_budget_graph(results_df)
            except Exception as e:
                print(f"Graph Error: {e}")

    return render_template('index.html', states=states, months=months, results=results, graph=graph, name=current_user.name)

# ==========================================
# CREATE TABLES AND RUN APP
# ==========================================
# Ye line check karegi ki 'user' table bani hai ya nahi. Agar nahi toh bana degi.
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)