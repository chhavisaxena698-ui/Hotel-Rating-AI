
import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Hotel Rating Prediction",
    page_icon="🏨",
    layout="wide"
)

# --------------------------------
# 2. LOAD DATASET AND MODEL
# --------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("tripadvisor_hotel_reviews.csv")

@st.cache_resource
def load_model():
    return joblib.load("hotel_rating_model.pkl")

data = load_data()
model = load_model()

# --------------------------------
# 3. DASHBOARD TITLE
# --------------------------------

st.title("🏨 Hotel Review Rating Prediction")
st.markdown(
    "### AI Powered Hotel Review Analysis Dashboard"
)

st.markdown(
    "Predict hotel ratings using Machine Learning "
    "based on customer reviews."
)

st.divider()

# --------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------

st.sidebar.title("🏨 Hotel Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "EDA Analysis",
        "Rating Prediction",
        "Raw Data"
    ]
)

# --------------------------------
# 5. OVERVIEW SECTION
# --------------------------------

if page == "Overview":

    st.header("Dashboard Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Reviews",
        f"{len(data):,}"
    )

    col2.metric(
        "Average Rating",
        f"{data['Rating'].mean():.2f} / 5"
    )

    col3.metric(
        "Highest Rating",
        f"{data['Rating'].max():.0f} / 5"
    )

    st.divider()

    st.subheader("About This Project")

    st.write(
        "This project uses Machine Learning and Natural "
        "Language Processing to analyse hotel reviews "
        "and predict customer ratings."
    )

    st.subheader("Project Technologies")

    st.write(
        "- Python\n"
        "- Pandas\n"
        "- Scikit-learn\n"
        "- TF-IDF Vectorization\n"
        "- Linear Regression\n"
        "- Streamlit"
    )

# --------------------------------
# 6. EDA ANALYSIS SECTION
# --------------------------------

elif page == "EDA Analysis":

    st.header("📊 Exploratory Data Analysis")

    st.subheader("Rating Distribution")

    rating_counts = (
        data["Rating"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(rating_counts)

    st.subheader("Rating Statistics")

    st.dataframe(
        data["Rating"].describe().to_frame(),
        use_container_width=True
    )

    st.subheader("Customer Review Length")

    review_lengths = data["Review"].astype(str).str.len()

    st.bar_chart(
        review_lengths.value_counts().head(15)
    )

# --------------------------------
# 7. RATING PREDICTION SECTION
# --------------------------------

elif page == "Rating Prediction":

    st.header("🤖 AI Rating Prediction")

    st.write(
        "Enter a hotel review below to predict its rating."
    )

    review = st.text_area(
        "Enter Customer Review",
        placeholder="Example: The hotel was excellent, "
                    "rooms were clean and staff was helpful."
    )

    if st.button("Predict Rating", type="primary"):

        if review.strip():

            prediction = model.predict([review])[0]

            prediction = max(1, min(5, prediction))

            st.success("Prediction Completed!")

            st.metric(
                "Predicted Hotel Rating",
                f"{prediction:.2f} / 5 ⭐"
            )

        else:

            st.warning(
                "Please enter a hotel review first."
            )

# --------------------------------
# 8. RAW DATA SECTION
# --------------------------------

elif page == "Raw Data":

    st.header("📁 Original Dataset")

    st.write(
        "Explore the original hotel reviews dataset."
    )

    st.dataframe(
        data,
        use_container_width=True,
        height=450
    )

    st.write(
        "Total Records:",
        len(data)
    )

# --------------------------------
# 9. FOOTER
# --------------------------------

st.divider()

st.caption(
    "Hotel Review Rating Prediction | "
    "Machine Learning Project"
)