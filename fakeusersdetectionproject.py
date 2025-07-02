import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tempfile
import os

# Set seed for reproducibility
np.random.seed(1)

lang_dict = {}  # to store globally for Gradio dropdown

# ---------- Helper Functions ----------

def process_data(df_users, df_fusers):
    df_users["isFake"] = 0
    df_fusers["isFake"] = 1

    df_all = pd.concat([df_users, df_fusers], ignore_index=True)
    df_all.columns = df_all.columns.str.strip()
    df_all = df_all.sample(frac=1).reset_index(drop=True)

    y = df_all["isFake"]
    x = df_all.drop(["isFake", "name"], axis=1)

    global lang_dict
    lang_list = list(enumerate(np.unique(x["lang"])))
    lang_dict = {name: i for i, name in lang_list}
    x["lang_num"] = x["lang"].map(lambda x: lang_dict.get(x, 0)).astype(int)
    x.drop(["lang"], axis=1, inplace=True)

    x = x.replace(np.nan, 0)

    features = [
        "statuses_count", "followers_count", "friends_count",
        "favourites_count", "lang_num", "listed_count",
        "geo_enabled", "profile_use_background_image"
    ]
    x = x[features]

    return x, y

def build_model(input_dim):
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def plot_metrics(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].plot(history.history['accuracy'], label='Train')
    axs[0].plot(history.history['val_accuracy'], label='Val')
    axs[0].set_title('Accuracy')
    axs[0].legend()
    
    axs[1].plot(history.history['loss'], label='Train')
    axs[1].plot(history.history['val_loss'], label='Val')
    axs[1].set_title('Loss')
    axs[1].legend()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(temp.name)
    plt.close()
    return temp.name

# ---------- Main Training Function ----------

def train_and_visualize(user_file, fake_file):
    try:
        df_users = pd.read_csv(user_file.name)
        df_fusers = pd.read_csv(fake_file.name)
        X, y = process_data(df_users, df_fusers)

        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state=0)
        train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, train_size=0.8, random_state=0)

        model = build_model(input_dim=8)
        history = model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=15, verbose=0)

        preds = (model.predict(test_X) > 0.5).astype("int32")
        acc = accuracy_score(test_y, preds)
        graph_path = plot_metrics(history)

        # Save model
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("model.weights.h5")

        return f"✅ Training Complete!\nTest Accuracy: {acc:.2f}", graph_path

    except Exception as e:
        return f"❌ Error: {str(e)}", None

# ---------- Prediction Function ----------

def predict_fake_user(statuses_count, followers_count, friends_count, favourites_count,
                      lang, listed_count, geo_enabled, profile_background):

    input_data = np.array([[ 
        statuses_count, followers_count, friends_count, favourites_count,
        lang_dict.get(lang, 0), listed_count, int(geo_enabled), int(profile_background)
    ]])

    # Load model
    with open("model.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.weights.h5")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    prediction = model.predict(input_data)[0][0]
    label = "Fake ❌" if prediction > 0.5 else "Not Fake ✅"
    return f"Probability: {prediction:.2f}", label

# ---------- Gradio Interface ----------

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>Fake User Detection Using ANN</h1>")

    with gr.Tab("1. Upload & Train Model"):
        user_file = gr.File(label="Upload Real Users CSV")
        fake_file = gr.File(label="Upload Fake Users CSV")
        output_msg = gr.Textbox(label="Status")
        output_img = gr.Image(label="Training Accuracy/Loss Graphs")

        train_btn = gr.Button("Start Training 🚀")
        train_btn.click(fn=train_and_visualize, inputs=[user_file, fake_file], outputs=[output_msg, output_img])

    with gr.Tab("2. Predict User"):
        statuses_count = gr.Number(label="Statuses Count (Number of posts or tweets)")
        followers_count = gr.Number(label="Followers Count (Number of followers)")
        friends_count = gr.Number(label="Friends Count (Number of people the user is following)")
        favourites_count = gr.Number(label="Favourites Count (Number of posts liked by the user)")
        language = gr.Dropdown(choices=["en"], label="Language (Current Language of the User's Account)")
        listed_count = gr.Number(label="Listed Count (Number of lists the user is included in)")
        geo_enabled = gr.Checkbox(label="Geo Enabled (Does the user share their location in posts?)")
        profile_background = gr.Checkbox(label="Profile Background (Does the user have a background image?)")
        pred_text = gr.Textbox(label="Prediction Probability")
        pred_label = gr.Textbox(label="Prediction Label")

        predict_btn = gr.Button("Predict User")
        predict_btn.click(
            fn=predict_fake_user,
            inputs=[ 
                statuses_count, followers_count, friends_count, favourites_count,
                language, listed_count, geo_enabled, profile_background
            ],
            outputs=[pred_text, pred_label]
        )

demo.launch()
