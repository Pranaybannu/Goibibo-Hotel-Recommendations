import pandas as pd
import numpy as np
import joblib
from flask import Flask,render_template_string,request,jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

##data loading
data = joblib.load('preprocessed_goibibo')
data = data.reset_index(drop=True)
tfidf = joblib.load('tfidf')
tfidf_matrix = joblib.load('tfidf_matrix')

##recommendation system

def recommend_hotels(city,property_type,room_type,star_rating,budget):
    filter_data = data[
        (data['city']==city)&
        (data['property_type']==property_type)&
        (data['room_type_cleaned']==room_type)
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


##FLASK APP
app = Flask(__name__)
input_row = 45
sample_data = data.iloc[input_row]

cities= sorted(data['city'].unique())
property_types = sorted(data['property_type'].unique())
room_types = sorted(data['room_type_cleaned'].unique())
star_ratings = sorted(data['hotel_star_rating'].unique())
budgets = sorted(data['price'].unique())

default_city = sample_data['city']
default_property= sample_data['property_type']
default_room = sample_data['room_type_cleaned']
default_rating = sample_data['hotel_star_rating']
default_budget = sample_data['price']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Hotel Recommendation System</title>

    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', sans-serif;
            min-height: 100vh;
            background:
                linear-gradient(rgba(15, 23, 42, 0.62), rgba(15, 23, 42, 0.72)),
                url('https://images.unsplash.com/photo-1566073771259-6a8506099945?auto=format&fit=crop&w=1600&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            padding: 32px;
            color: #111827;
        }

        .hero {
            max-width: 900px;
            margin: 0 auto 30px;
            text-align: center;
            color: white;
        }

        .hero h1 {
            font-size: 44px;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .hero p {
            font-size: 18px;
            opacity: 0.95;
        }

        .form-container,
        #recommendations {
            max-width: 1200px;
            margin: auto;
        }

        .form-container {
            background: rgba(255, 255, 255, 0.97);
            padding: 34px;
            border-radius: 22px;
            box-shadow: 0 24px 70px rgba(0, 0, 0, 0.22);
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-weight: 700;
            margin-bottom: 8px;
            color: #374151;
        }

        select,
        input {
            width: 100%;
            padding: 14px 15px;
            border-radius: 12px;
            border: 1px solid #d1d5db;
            font-size: 15px;
            background: white;
        }

        select:focus,
        input:focus {
            outline: none;
            border-color: #2563eb;
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14);
        }

        button {
            width: 100%;
            margin-top: 26px;
            padding: 17px;
            border: none;
            border-radius: 14px;
            background: linear-gradient(135deg, #2563eb, #059669);
            color: white;
            font-size: 18px;
            font-weight: 800;
            cursor: pointer;
            transition: 0.25s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 30px rgba(37, 99, 235, 0.32);
        }

        .loading {
            margin-top: 28px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: 800;
            padding: 36px;
        }

        .hotel-card {
            background: rgba(255, 255, 255, 0.97);
            margin-top: 28px;
            border-radius: 20px;
            padding: 26px;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.16);
        }

        .hotel-banner {
            display: inline-block;
            background: #059669;
            color: white;
            padding: 8px 15px;
            border-radius: 999px;
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 16px;
        }

        .hotel-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 18px;
            flex-wrap: wrap;
        }

        .hotel-name {
            font-size: 28px;
            font-weight: 800;
        }

        .hotel-location {
            color: #6b7280;
            margin-top: 6px;
        }

        .price-box {
            background: #059669;
            color: white;
            padding: 13px 22px;
            border-radius: 14px;
            font-size: 23px;
            font-weight: 800;
            text-align: center;
            min-width: 130px;
        }

        .price-night {
            font-size: 12px;
            font-weight: 600;
        }

        .hotel-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 22px;
        }

        .hotel-tags span {
            background: #eff6ff;
            color: #1d4ed8;
            padding: 8px 14px;
            border-radius: 999px;
            font-size: 14px;
            font-weight: 700;
        }

        .similarity-score {
            margin-top: 18px;
            padding: 13px;
            background: #ecfdf5;
            color: #047857;
            border-radius: 12px;
            font-weight: 800;
        }

        .details-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 18px;
            margin-top: 24px;
        }

        .detail-card {
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 18px;
        }

        .detail-card h3 {
            margin-bottom: 10px;
            font-size: 17px;
        }

        .detail-card p {
            color: #4b5563;
            line-height: 1.5;
            font-size: 14px;
        }

        .no-results {
            background: white;
            border-radius: 20px;
            padding: 42px;
            margin-top: 30px;
            text-align: center;
            box-shadow: 0 16px 42px rgba(0, 0, 0, 0.14);
        }

        @media (max-width: 768px) {
            body {
                padding: 20px;
            }

            .hero h1 {
                font-size: 32px;
            }

            .form-container {
                padding: 24px;
            }

            .price-box {
                width: 100%;
            }
        }
    </style>
</head>

<body>

    <section class="hero">
        <h1>AI Hotel Recommendation System</h1>
        <p>Find the best hotels using smart machine learning recommendations</p>
    </section>

    <main>
        <section class="form-container">
            <form id="recommendForm">

                <div class="form-grid">

                    <div class="input-group">
                        <label for="city">City</label>
                        <select id="city" name="city">
                            {% for city in cities %}
                            <option value="{{ city }}" {% if city == default_city %}selected{% endif %}>
                                {{ city }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="property_type">Property Type</label>
                        <select id="property_type" name="property_type">
                            {% for property in property_types %}
                            <option value="{{ property }}" {% if property == default_property %}selected{% endif %}>
                                {{ property }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="room_type">Room Type</label>
                        <select id="room_type" name="room_type">
                            {% for room in room_types %}
                            <option value="{{ room }}" {% if room == default_room %}selected{% endif %}>
                                {{ room }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="star_rating">Star Rating</label>
                        <select id="star_rating" name="star_rating">
                            {% for rating in star_ratings %}
                            <option value="{{ rating }}" {% if rating == default_rating %}selected{% endif %}>
                                {{ rating }} Star
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div class="input-group">
                        <label for="budget">Budget Per Night</label>
                        <input
                            id="budget"
                            type="number"
                            name="budget"
                            value="{{ default_budget }}"
                            min="0">
                    </div>

                </div>

                <button type="submit">Search Hotels</button>

            </form>
        </section>

        <section id="recommendations"></section>
    </main>

    <script>
        function fillDropdown(dropdownId, values, selectedValue = "") {
            const dropdown = document.getElementById(dropdownId);
            dropdown.innerHTML = "";

            values.forEach(value => {
                const option = document.createElement("option");
                option.value = value;
                option.textContent = value;

                if (String(value) === String(selectedValue)) {
                    option.selected = true;
                }

                dropdown.appendChild(option);
            });
        }

        async function updateOptions(changedField) {
            const city = document.getElementById("city").value;
            const propertyType = document.getElementById("property_type").value;
            const roomType = document.getElementById("room_type").value;

            const payload = {
                city: city,
                property_type: changedField === "city" ? "" : propertyType,
                room_type: changedField === "city" || changedField === "property_type" ? "" : roomType
            };

            const response = await fetch("/get_options", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (changedField === "city") {
                fillDropdown("property_type", result.property_types || []);
                fillDropdown("room_type", result.room_types || []);
            }

            if (changedField === "property_type") {
                fillDropdown("room_type", result.room_types || []);
            }

            fillDropdown("star_rating", result.star_ratings || []);

            if (result.max_budget && result.max_budget > 0) {
                document.getElementById("budget").max = result.max_budget;
                document.getElementById("budget").placeholder =
                    "Budget between ₹" + result.min_budget + " and ₹" + result.max_budget;
            }
        }

        document.addEventListener("DOMContentLoaded", function () {
            document.getElementById("city").addEventListener("change", function () {
                updateOptions("city");
            });

            document.getElementById("property_type").addEventListener("change", function () {
                updateOptions("property_type");
            });

            document.getElementById("room_type").addEventListener("change", function () {
                updateOptions("room_type");
            });

            document.getElementById("recommendForm").addEventListener("submit", async function(e) {
                e.preventDefault();

                document.getElementById("recommendations").innerHTML = `
                    <div class="loading">
                        Finding the best hotels for you...
                    </div>
                `;

                const formData = new FormData(this);

                const response = await fetch("/recommend", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                let html = "";

                if (data.length === 0) {
                    html = `
                        <div class="no-results">
                            <h2>No Hotels Found</h2>
                            <p>Try increasing budget, changing room type, or reducing star rating.</p>
                        </div>
                    `;
                } else {
                    data.forEach(hotel => {
                        html += `
                            <div class="hotel-card">
                                <div class="hotel-banner">Recommended</div>

                                <div class="hotel-header">
                                    <div>
                                        <div class="hotel-name">${hotel.property_name}</div>
                                        <div class="hotel-location">${hotel.city}</div>
                                    </div>

                                    <div class="price-box">
                                        ₹ ${hotel.price}
                                        <br>
                                        <span class="price-night">per night</span>
                                    </div>
                                </div>

                                <div class="hotel-tags">
                                    <span>${hotel.property_type}</span>
                                    <span>${hotel.room_type_cleaned}</span>
                                    <span>${hotel.hotel_star_rating} Star</span>
                                </div>

                                <div class="similarity-score">
                                    Match Score: ${(hotel.similarity_score * 100).toFixed(1)}%
                                </div>

                                <div class="details-grid">
                                    <div class="detail-card">
                                        <h3>Address</h3>
                                        <p>${hotel.address}</p>
                                    </div>

                                    <div class="detail-card">
                                        <h3>Facilities</h3>
                                        <p>${hotel.hotel_facilities}</p>
                                    </div>

                                    <div class="detail-card">
                                        <h3>Nearby Attractions</h3>
                                        <p>${hotel.point_of_interest}</p>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                }

                document.getElementById("recommendations").innerHTML = html;
            });
        });
    </script>

</body>
</html>
"""

##home page  route
@app.route('/',methods=['GET','POST'])
def home():
    return render_template_string(HTML_TEMPLATE,
                                  cities=cities,
                                  property_types = property_types,
                                  room_types=room_types,
                                  star_ratings = star_ratings,
                                  default_city = default_city,
                                  default_property_type = default_property,
                                  default_room_type = default_room,
                                  default_rating = default_rating)

##route to get options for dropdowns
@app.route('/get_options',methods=['POST'])
def get_options():
    payload = request.get_json() or {}
    city = payload.get('city')
    property_type = payload.get('property_type')
    room_type = payload.get('room_type')

    filtered = data.copy()

    if city:
        filtered = filtered[filtered['city']==city]

    if property_type:
        filtered = filtered[filtered['property_type']==property_type]
    
    if room_type:
        filtered = filtered[filtered['room_type_cleaned']==room_type]
    
    cities = sorted(filtered['city'].unique().tolist())
    property_types = sorted(filtered['property_type'].unique().tolist())
    room_types = sorted(filtered['room_type_cleaned'].unique().tolist())

    return jsonify({'cities':cities,
                    'property_types':property_types,
                    'room_types':room_types,
                    'star_ratings':sorted(filtered['hotel_star_rating'].unique().tolist()),
                    'min_budget': int(filtered['price'].min() if not filtered.empty else 0),
                    'max_budget': int(filtered['price'].max() if not filtered.empty else 0)
                    })

##route for hotel recommendations
@app.route('/recommend',methods=['POST'])
def recommend():
    city = request.form.get('city')
    property_type = request.form.get('property_type')
    room_type = request.form.get('room_type')
    star_rating = int(request.form.get('star_rating',0))
    budget = int(request.form.get('budget',0))

    recommendations = recommend_hotels(city,property_type,room_type,star_rating,budget)
    if recommendations.empty:
        return jsonify([])
    
    cols = ['property_name','city','property_type', 'room_type_cleaned','hotel_star_rating',
             'hotel_facilities', 'address', 'point_of_interest', 'price','similarity_score']
    
    recommendations = recommendations[cols].to_dict(orient='records')
    print(recommendations)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8000,debug=True)

