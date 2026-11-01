import gradio as gr
import joblib
import os

# Load Trained Machine Learning Model
model = joblib.load("hotel_rating_model.pkl")


# Function to Predict Hotel Rating
def predict_rating(review):

    if not review or not review.strip():
        return "Please enter a hotel review."

    prediction = model.predict([review])[0]

    prediction = max(1, min(5, prediction))

    return f"{prediction:.2f} / 5 Stars"


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
        # 🏨 Hotel Review Rating Prediction

        ### AI Powered Hotel Rating System

        Enter your hotel experience below.
        Our Machine Learning Model will predict
        the expected hotel rating based on your review.
        """
    )

    gr.Markdown("---")

    review_input = gr.Textbox(
        label="Enter Your Hotel Review",
        placeholder="Write your hotel experience here...",
        lines=5
    )

    predict_button = gr.Button(
        "Predict Hotel Rating ⭐",
        variant="primary"
    )

    output = gr.Textbox(
        label="Predicted Hotel Rating",
        interactive=False
    )

    predict_button.click(
        fn=predict_rating,
        inputs=review_input,
        outputs=output
    )

    gr.Examples(
        examples=[
            ["Excellent hotel with clean rooms and helpful staff."],
            ["Very poor service and dirty rooms."],
            ["Average hotel with decent food and facilities."],
            ["Amazing experience, friendly staff and delicious food."]
        ],
        inputs=review_input
    )

    gr.Markdown(
        """
        ---
        **Project:** Hotel Review Rating Prediction

        **Technology:** Python, TF-IDF, Linear Regression and Gradio

        **Developed as an AI & Machine Learning Project**
        """
    )


# Launch Gradio Application
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)))