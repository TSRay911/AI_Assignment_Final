import streamlit as st
import pandas as pd
import torch
from models.model import MatrixFactorization
from models.model import NeuralNet
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


st.set_page_config(layout="wide")
st.title('Anime Recommender System')

# Initializing variables
with open('models/mappings.pkl','rb') as f:
    mappings = pickle.load(f)

userid2idx = mappings['userid2idx']
animeid2idx = mappings['animeid2idx']
idx2uanimeid = mappings['idx2uanimeid']

anime_df = pd.read_parquet('dataset/cleaned_animes.parquet')
anime_df = anime_df.reset_index(drop=True)

anime_names = anime_df.set_index('anime_id')['Name'].to_dict()
name_to_id = {name: aid for aid, name in anime_names.items()}

# Initialize the model
model = MatrixFactorization(268664, 13125, n_factors=32)
model.load_state_dict(torch.load('models/collaborative_weights.pth', map_location=torch.device('cpu')))
model.eval()

net = NeuralNet()
net.load_state_dict(torch.load('models/trained_image_model_SGDV3.pth', map_location=torch.device('cpu')))
net.eval()

new_transform = transforms.Compose([
    transforms.Resize(256),            
    transforms.CenterCrop(224), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

class_name = ['action','horror','romance','sci-fi','sports']

@st.dialog("Anime Details", width="large")
def anime_details(item):
    col1, col2 = st.columns([1, 2])  

    with col1:
        st.image(item['poster_url'], width=250)

        st.markdown(f"""
            <div style="text-align: center; margin-top: 10px;">
                <h1 style="margin-bottom: 5px; max-width: 250px; word-wrap: break-word; overflow-wrap: break-word;">
                    {item['anime_name']}
                </h1>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="text-align: left;">
                <p style="font-size: 14px; font-weight: bold; color: #d1d1d1; 
                    background-color: #333; display: inline-block; padding: 5px 15px; 
                    border-radius: 8px; width:100%; text-align:center;">
                    {item['anime_genres']}
                </p>
                <p style="text-align: justify; font-size: 16px; line-height: 1.6; margin-top: 15px;">
                    {item['anime_synopsis']}
                </p>
            </div>
        """, unsafe_allow_html=True)

# Display the shows function
def display_results(recommendations):

    cols = st.columns(6)  
    for i, show in enumerate(recommendations):

        anime_info = anime_df.loc[anime_df['Name'] == show].iloc[0]

        # Store relevant details
        anime_data = {
            "anime_name": anime_info["Name"],
            "poster_url": anime_info["Image URL"],
            "anime_score": anime_info["Score"],
            "anime_genres": anime_info["Genres"],
            "anime_synopsis": anime_info["Synopsis"],
        }

        html_code = f"""
        <div style="width:100%; height:350px; overflow:hidden; margin-bottom: 10px;">
            <img src="{anime_data['poster_url']}" style="width:100%; height:100%; object-fit:cover; border-radius:10px;">
                    
        </div>
        <div style="width:100%; height:100px; display:flex; align-items:center; justify-content:center; margin-bottom:10px;">
            <p style="margin:0; font-size:16px; text-align:center;">{anime_data['anime_name']}</p>
        </div>
        """
        
        with cols[i % 6]:
            st.markdown(html_code, unsafe_allow_html=True)
            
            if st.button("More Details", key=f'btn{i}'):
                anime_details(anime_data)

# Recommend shows based on similarity function
def recommend_anime_content(selected_anime, num_of_shows):

    synopsis = anime_df['Synopsis'].fillna('')
    genres = anime_df['Genres'].fillna('')
    name = anime_df['Name'].fillna('')      

    synopsis_vectorizer = TfidfVectorizer(stop_words='english')
    genres_vectorizer = TfidfVectorizer(stop_words='english')
    name_vectorizer = TfidfVectorizer(stop_words=None)

    tfidf_synopsis = synopsis_vectorizer.fit_transform(synopsis)
    tfidf_genres = genres_vectorizer.fit_transform(genres)
    tfidf_name = name_vectorizer.fit_transform(name)


    if selected_anime not in name_to_id:
        st.error("Show not found in the dataset.")
        return []
    
    else:
        idx = anime_df[anime_df['Name'] == selected_anime].index[0]
    
    cosine_sim_synopsis = cosine_similarity(tfidf_synopsis[idx], tfidf_synopsis).flatten()
    cosine_sim_genres = cosine_similarity(tfidf_genres[idx], tfidf_genres).flatten()
    cosine_sim_name = cosine_similarity(tfidf_name[idx], tfidf_name).flatten()

    combined_sim = cosine_sim_name * 2 + cosine_sim_genres * 4 + cosine_sim_synopsis * 4

    similar_indices = combined_sim.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:num_of_shows]

    recommendations = anime_df.iloc[similar_indices]['Name'].tolist()
    return recommendations

# Recommend shows using collaborative algorithm function
def recommend_anime_collaborative(selected_anime, model, num_of_shows):
    
    if selected_anime not in name_to_id:
        st.error("Show not found in the dataset.")
        return []
    
    selected_anime_id = name_to_id[selected_anime]
    
    if selected_anime_id not in animeid2idx:
        st.error("Anime ID not found in mapping.")
        return []
    
    anime_idx = animeid2idx[selected_anime_id]
    selected_embedding = model.item_factors.weight[anime_idx]
    
    all_embeddings = model.item_factors.weight
    
    similarities = torch.matmul(all_embeddings, selected_embedding)
    similarities = similarities.detach().cpu().numpy()
    
    sorted_indices = similarities.argsort()[::-1]
    recommended_indices = [idx for idx in sorted_indices if idx != anime_idx][:num_of_shows]
    
    recommended_anime_ids = [idx2uanimeid[idx] for idx in recommended_indices]
    recommended_shows = [anime_names.get(aid, "Unknown") for aid in recommended_anime_ids]
    
    return recommended_shows

def recommend_anime_image(uploaded_image, num_of_shows):

    image = Image.open(uploaded_image)
    image = new_transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
        predicted_genre = class_name[predicted.item()]
    
    st.write(f"### The image you've uploaded is similar to the {predicted_genre} genre.")
    filtered_df = anime_df[anime_df['Genres'].str.contains(predicted_genre, case=False, na=False)].sort_values(by="Score",ascending=False)
    recommendations = filtered_df['Name'].head(num_of_shows).tolist()
    return recommendations

# Sidebar 
with st.sidebar:
    system_option = st.radio(
        "Choose A Recommender Algorithm",
        ("Content-Based Filtering", 
         "Collaborative Filtering", 
         "Image-Based Filtering")
    )

    num_of_shows = st.number_input("Number of shows", min_value=6, step=6)

# Give options to upload image if user choose image based recommendation system
if system_option == "Content-Based Filtering" or system_option == "Collaborative Filtering":
    selected_anime = st.selectbox("Select a show:", anime_df['Name'])
else:
    uploaded_image = st.file_uploader("Choose a file", type=['jpg','png','jpeg'])



if system_option == "Collaborative Filtering":
    st.write(f"### Users who like {selected_anime} also like:")
    recommendations = recommend_anime_collaborative(selected_anime, model, num_of_shows)
    display_results(recommendations)

elif system_option == "Content-Based Filtering":
    st.write(f"### Shows that are similar to {selected_anime}:")
    recommendations = recommend_anime_content(selected_anime, num_of_shows)
    display_results(recommendations)

elif system_option == "Image-Based Filtering" and uploaded_image is not None:
    recommendations = recommend_anime_image(uploaded_image, num_of_shows)
    display_results(recommendations)


# CSS code
st.markdown(
    """
    <style>
        .stRadio > label div p{
            font-size: 20px;
            font-weight: bold;
        }

        div[data-testid="stButton"] button {
            font-size: 18px;
            padding: 12px 24px;
            width: 100%;
        }

        .st-bp{
            justify-content:center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

