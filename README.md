# Goibibo Hotel Recommendation System

## 1. Project Overview

This project is a **content-based hotel recommendation system** built from Goibibo hotel-listing data. It prepares hotel descriptions, facilities, room information, property details, and star ratings for text-based retrieval, then recommends relevant hotels using **TF-IDF vectorisation** and **cosine similarity**.

The system returns up to **10 ranked hotel recommendations** based on a traveller’s preferences.

The project includes:

1. A **Streamlit application** for an interactive recommendation interface.
2. A **Flask application** with a web interface and recommendation workflow.
3. A **Jupyter Notebook** for preprocessing, feature engineering, TF-IDF vectorisation, and recommendation logic.

---

## 2. Project Objectives

The project is designed to:

* Clean and standardise raw Goibibo hotel-listing data.
* Handle missing hotel-description and facility information.
* Convert inconsistent room-type labels into standard categories.
* Standardise property types.
* Create text features for each hotel.
* Transform hotel text into TF-IDF vectors.
* Filter hotels using traveller preferences.
* Rank eligible hotels using cosine similarity.
* Display the top hotel recommendations through Streamlit and Flask applications.

---

## 3. Recommendation Workflow

```text
Raw Goibibo hotel-listing data
        │
        ▼
Data cleaning and column selection
        │
        ├── Remove unused columns
        ├── Handle missing text fields
        ├── Standardise room types
        └── Standardise property types
        │
        ▼
Synthetic price generation
        │
        ▼
Hotel feature-text creation
        │
        ├── Property name
        ├── Property type
        ├── Room type
        ├── Hotel facilities
        ├── Room facilities
        ├── Star rating
        ├── Hotel category
        ├── Hotel description
        └── Additional information
        │
        ▼
TF-IDF vectorisation
        │
        ▼
Traveller preference input
        │
        ├── City
        ├── Property type
        ├── Room type
        ├── Star rating
        └── Budget
        │
        ▼
Candidate hotel filtering
        │
        ▼
Cosine-similarity ranking
        │
        ▼
Top 10 hotel recommendations
```

---

## 4. Dataset Overview

The project uses a Goibibo hotel-travel sample containing **4,000 rows** and **36 columns**.

### 4.1 Dataset Categories

| Category           | Example columns                                                                 |
| ------------------ | ------------------------------------------------------------------------------- |
| Hotel identity     | `property_name`, `property_id`, `uniq_id`, `hotel_brand`                        |
| Location           | `city`, `address`, `area`, `locality`, `state`, `country`                       |
| Property details   | `property_type`, `hotel_category`, `hotel_star_rating`                          |
| Room details       | `room_type`, `room_facilities`, `room_area`, `room_count`                       |
| Hotel information  | `hotel_description`, `additional_info`, `hotel_facilities`, `point_of_interest` |
| Review information | `guest_recommendation`, `site_review_count`, `site_review_rating`               |
| Other metadata     | `latitude`, `longitude`, `pageurl`, `crawl_date`                                |

---

## 5. Data Cleaning and Preparation

### 5.1 Unused Columns Removed

The notebook removes fields that are not required for the recommendation workflow, including:

* `similar_hotel`
* `site_review_count`
* `site_review_rating`
* `site_stay_review_rating`
* `sitename`
* `state`
* `qts`
* `query_time_stamp`
* `review_count_by_category`
* `room_area`
* `room_count`
* `image_count`
* `latitude`
* `longitude`
* `pageurl`
* `area`
* `country`
* `crawl_date`
* `guest_recommendation`
* `hotel_brand`
* `locality`
* `property_id`
* `province`

### 5.2 Missing-Value Handling

The following text fields are filled with fallback values:

| Field               | Replacement value     |
| ------------------- | --------------------- |
| `additional_info`   | `not available`       |
| `hotel_description` | `not available`       |
| `hotel_facilities`  | `not available`       |
| `point_of_interest` | `details unavailable` |
| `room_facilities`   | `details unavailable` |

This ensures that incomplete records can still be represented as text and included in the recommendation process.

---

## 6. Room-Type and Property-Type Standardisation

### 6.1 Room-Type Cleaning

The notebook retains the 20 most frequent room types and maps inconsistent labels into common categories.

| Raw room type examples                                 | Cleaned room type |
| ------------------------------------------------------ | ----------------- |
| Standard Room, Standard, AC Room, Double Room          | Standard          |
| Deluxe Room, Deluxe, Super Deluxe Room, Deluxe AC Room | Deluxe            |
| Superior Room                                          | Superior          |
| Executive Room                                         | Executive         |
| Suite Room                                             | Suite             |
| Premium Room, Luxury Room                              | Luxury            |

The cleaned room category is stored in:

```python
room_type_cleaned
```

### 6.2 Property-Type Cleaning

Property types are converted into title case.

The following property types are grouped under **Bnb**:

* Farm Stay
* Motel
* Hostel
* Tent

---

## 7. Synthetic Price Generation

The raw dataset used in the notebook does not provide a price field for the recommendation workflow. Therefore, the project creates a **synthetic price** using:

1. A base price based on property type.
2. An additional charge based on room type.
3. A random amount between `-500` and `2000`.

### Base Prices

| Property type     | Base price |
| ----------------- | ---------: |
| Lodge             |      1,500 |
| Bnb               |      2,000 |
| Guest House       |      2,500 |
| Homestay          |      2,800 |
| Service Apartment |      3,000 |
| Hotel             |      3,500 |
| Houseboat         |      4,000 |
| Cottage           |      4,500 |
| Resort            |      6,000 |
| Villa             |      8,000 |
| Bungalow          |     10,000 |
| Palace            |     15,000 |

> **Important:** The `price` column is generated only for demonstration and budget filtering. It is not a live, observed, or bookable Goibibo price.

---

## 8. Hotel Feature Engineering

Each hotel is converted into a combined text field named `hotel_features`.

The feature text contains:

* Property name
* Property type
* Cleaned room type
* Room facilities
* Hotel star rating
* Hotel category
* Hotel facilities
* Hotel description
* Additional information

Example logic:

```python
data['hotel_features'] = (
    data['property_name'] + ' ' +
    data['property_type'] + ' ' +
    data['room_type_cleaned'] + ' ' +
    data['room_facilities'] + ' ' +
    data['hotel_star_rating'].fillna('').astype(str) + ' ' +
    data['hotel_category'] + ' ' +
    data['hotel_facilities'] + ' ' +
    data['hotel_description'] + ' ' +
    data['additional_info']
)
```

This consolidated field allows the recommender to compare hotel details with the traveller’s query.

---

## 9. Recommendation Method

### 9.1 TF-IDF Vectorisation

The project uses `TfidfVectorizer` from scikit-learn.

```python
tfidf = TfidfVectorizer(stop_words='english')
tf_matrix = tfidf.fit_transform(data['hotel_features'])
```

TF-IDF converts each hotel’s text features into a numerical vector. Words that help distinguish a hotel receive more importance than commonly occurring words.

### 9.2 Candidate Filtering

The notebook filters hotels using:

* City
* Property type
* Cleaned room type
* Budget
* Minimum star rating

```python
filter_data = data[
    (data['city'] == city) &
    (data['property_type'] == property_type) &
    (data['room_type_cleaned'] == room_type) &
    (data['price'] <= budget) &
    (data['hotel_star_rating'] >= star_rating)
]
```

If no eligible hotel is found, the function returns:

```python
"no hotels found"
```

### 9.3 Cosine Similarity Ranking

The system creates a query from the user preferences:

```python
query = f"{city} {property_type} {room_type} {budget} {star_rating} star hotel"
```

It transforms the query using the same fitted TF-IDF vectoriser:

```python
query_vector = tfidf.transform([query])
```

Then, it calculates cosine similarity between the query and the eligible hotel vectors:

```python
similarity_score = cosine_similarity(query_vector, tf_matrix[indices])
```

The hotels are ranked by similarity score and the top 10 are returned.

```python
recommended = filter_data.sort_values(
    by='similarity_score',
    ascending=False
)

return recommended.head(10)
```

---

## 10. Saved Model Artifacts

The notebook saves the following reusable artifacts using Joblib.

| Artifact               | Purpose                               |
| ---------------------- | ------------------------------------- |
| `preprocessed_goibibo` | Cleaned and transformed hotel dataset |
| `tfidf`                | Fitted TF-IDF vectoriser              |
| `tfidf_matrix`         | TF-IDF matrix of hotel feature text   |

These files are loaded by the Streamlit and Flask applications so that preprocessing and vectorisation do not need to be repeated for every user request.

---

## 11. Streamlit Application Workflow

The Streamlit application provides cascading filters.

### User Flow

1. Select a city.
2. Select a property type available in that city.
3. Select a room type available for that city and property type.
4. Select a star rating.
5. Select a budget using a slider.
6. Click **Recommend Hotels**.

The application displays up to 10 recommended hotels with:

* Property name
* Address
* Nearby places
* Hotel facilities
* Price
* Similarity score

### Run Streamlit App

```bash
cd apps
streamlit run streamlithrapp.py
```

---

## 12. Flask Application Workflow

The Flask application loads the processed dataset and TF-IDF artifacts, then provides a web-based hotel recommendation interface.

It accepts:

* City
* Property type
* Room type
* Star rating
* Budget

The application:

1. Creates dropdown options from the processed dataset.
2. Filters hotel candidates.
3. Converts the preference query into a TF-IDF vector.
4. Calculates cosine similarity.
5. Returns the top 10 recommended hotels.

### Run Flask App

```bash
cd apps
python flaskhrapp.py
```

> **Implementation note:** The Flask application filters by city, property type, and room type, but does not strictly apply budget or star rating before ranking. The notebook and Streamlit workflow apply stricter filtering.

---

## 13. Repository Structure

```text
Goibibo-Hotel-Recommendations/
│
├── datasets/
│   ├── goibibo_com-travel_sample.csv
│   └── preprocessed_goibibo
│
├── notebook and others/
│   ├── HotelRecommendations.ipynb
│   ├── tfidf
│   └── tfidf_matrix
│
├── apps/
│   ├── streamlithrapp.py
│   ├── flaskhrapp.py
│   └── requirements.txt
│
└── README.md
```

---

## 14. Tools and Technologies

| Category                | Tools                            |
| ----------------------- | -------------------------------- |
| Data preparation        | Pandas, NumPy                    |
| Text feature extraction | scikit-learn `TfidfVectorizer`   |
| Similarity ranking      | scikit-learn `cosine_similarity` |
| Model-artifact storage  | Joblib                           |
| Interactive interface   | Streamlit                        |
| Web application         | Flask                            |
| Notebook visualisation  | Matplotlib, Seaborn              |
| Development environment | Google Colab / Jupyter Notebook  |

---

## 15. Installation

### Clone the Repository

```bash
git clone https://github.com/Pranaybannu/Goibibo-Hotel-Recommendations.git
cd Goibibo-Hotel-Recommendations
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib flask streamlit matplotlib seaborn
```

### Prepare Artifacts

The application scripts expect these files:

```text
preprocessed_goibibo
tfidf
tfidf_matrix
```

Copy them from the `notebook and others` folder into the application working directory, or update the file paths in the application scripts.

---

## 16. Limitations

1. Prices are synthetic and are not live booking prices.
2. The source dataset is static and does not represent real-time hotel availability.
3. Recommendations are content-based; they do not use booking history, user clicks, or collaborative filtering.
4. The similarity score represents text-feature similarity, not hotel quality or customer satisfaction.
5. Streamlit and Flask use slightly different filtering logic.
6. The saved preprocessing and TF-IDF artifacts must be available at the expected paths.

---

## 17. Future Improvements

1. Use real-time hotel pricing and availability data.
2. Add filters for amenities, location, guest rating, and distance from points of interest.
3. Build collaborative filtering using user interactions and bookings.
4. Create one common recommendation service for both Streamlit and Flask.
5. Add unit tests for preprocessing and recommendation functions.
6. Use environment variables or configuration files for artifact paths.
