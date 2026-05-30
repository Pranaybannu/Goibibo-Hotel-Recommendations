import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib
import html
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

##data loading
data = joblib.load('preprocessed_goibibo')
data = data.reset_index(drop=True)
tfidf = joblib.load('tfidf')
tfidf_matrix = joblib.load('tfidf_matrix')

##function to recommend hotels based on cosine similarity
def recommend_hotels(city,property_type,room_type,star_rating,budget):
    filter_data = data[
        (data['city']==city)&
        (data['property_type']==property_type)&
        (data['room_type_cleaned']==room_type)&
        (data['price']<=budget)
    ].copy()

    if city:
        city_filter = filter_data[filter_data['city']==city]

        if not city_filter.empty:
            filter_data = city_filter
    
    if property_type:
        property_filter = filter_data[filter_data['property_type']==property_type]

        if not property_filter.empty:
            filter_data = property_filter
    
    if room_type:
        room_filter = filter_data[filter_data['room_type_cleaned']==room_type]

        if not room_filter.empty:
            filter_data = room_filter
    
    if filter_data.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no hotels match the criteria
    
    query = f"""{city} {property_type} {room_type} {star_rating} {budget}"""
    query_vec = tfidf.transform([query])
    indices = filter_data.index
    filter_data = filter_data.drop(['uniq_id','room_type'],axis=1)
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix[indices])
    filter_data['similarity_score'] = cosine_sim.flatten()
    recommended = filter_data.sort_values(by='similarity_score', ascending=False)
    
    print(f"Found {len(recommended)} matching hotels")
    print("Returning top hotels based on similarity score")
    
    return recommended.head(10)

##streamlit app
st.set_page_config(page_title="Hotel Recommendation System",page_icon="🏨",layout="wide")
st.title("Hotel Recommendation System")

##city filters
city = st.selectbox("Select City", options=sorted(data['city'].unique()))
filtered_data = data[data['city']==city]

##property type filters
property_type = st.selectbox("Select Property Type", options=filtered_data['property_type'].unique())
filtered_data = filtered_data[filtered_data['property_type']==property_type]

##room type filters
room_type = st.selectbox("Select Room Type", options=filtered_data['room_type_cleaned'].unique())
filtered_data = filtered_data[filtered_data['room_type_cleaned']==room_type]

##star rating filters
min_star = int(filtered_data['hotel_star_rating'].min())
max_star = int(filtered_data['hotel_star_rating'].max())
if min_star == max_star:
    star_rating = min_star
    st.write(f"Only {min_star}-star hotels available in {city} for the selected property and room type.")
else:
    star_rating = st.selectbox("Select Star Rating", options=sorted(filtered_data['hotel_star_rating'].unique()))
    filtered_data = filtered_data[filtered_data['hotel_star_rating']==star_rating]


##budget filters
min_price = int(filtered_data['price'].min())
max_price = int(filtered_data['price'].max())
if min_price == max_price:
    budget = min_price
    st.write(f"Only hotels with price ₹{min_price} available in {city} for the selected property, room type and star rating.")
else:
    budget = st.slider("Select Budget", min_value=filtered_data['price'].min(), max_value=filtered_data['price'].max(), step=500)

        
##recommend hotels button        
if st.button("Recommend Hotels"):
    recommendations = recommend_hotels(city, property_type, room_type, star_rating, budget)
    
    if recommendations.empty:
        st.write("No hotels found matching the criteria.")
    else:
        cols = ['property_name','hotel_facilities', 'address', 'point_of_interest', 'price','similarity_score']
        recommendations = recommendations[cols]

        st.success("Hotels recommended successfully!")
        st.write(f"Found {len(recommendations)} hotels matching the criteria.")
        st.subheader("Top Recommended Hotels:")

        cards_html = """
        <style>
        .hotel-card {
            padding: 1.3rem;
            border-radius: 14px;
            background: #ffffff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
            font-family: Arial, sans-serif;
        }

        .hotel-name {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.4rem;
        }

        .hotel-address {
            font-size: 0.95rem;
            color: #64748b;
            margin-bottom: 0.8rem;
        }

        .hotel-row {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 0.7rem;
        }

        .hotel-badge {
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #334155;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .hotel-price {
            color: #15803d;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .hotel-section-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: #475569;
            margin-top: 0.7rem;
            margin-bottom: 0.25rem;
        }

        .hotel-text {
            font-size: 0.92rem;
            color: #334155;
            line-height: 1.5;
        }

        .score {
            color: #2563eb;
            font-weight: 700;
        }
        </style>
        """

        for _, hotel in recommendations.iterrows():
            property_name = html.escape(str(hotel['property_name']))
            address = html.escape(str(hotel['address']))
            sight_seeings = html.escape(str(hotel['point_of_interest']))
            facilities = html.escape(str(hotel['hotel_facilities']))

            cards_html += f"""
            <div class="hotel-card">
                <div class="hotel-name">{property_name}</div>
                <div class="hotel-address">{address}</div>

                <div class="hotel-row">
                    <span class="hotel-badge hotel-price">₹{int(hotel['price'])}</span>
                    <span class="hotel-badge">
                        Match Score:
                        <span class="score">{hotel['similarity_score']:.2f}</span>
                    </span>
                </div>

                <div class="hotel-section-title">Nearby Places</div>
                <div class="hotel-text">{sight_seeings}</div>

                <div class="hotel-section-title">Facilities</div>
                <div class="hotel-text">{facilities}</div>
            </div>
            """

        components.html(cards_html, height=900, scrolling=True)
                
