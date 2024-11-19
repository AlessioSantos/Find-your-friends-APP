import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px 

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.xlsx'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

def reset_button():
    for key in st.session_state.keys():
        del st.session_state[key]  

if "name" not in st.session_state:
    st.session_state["name"] = ""

with st.sidebar:
    if st.button("Resetuj"):
        reset_button()

    st.session_state["name"] = st.text_input("Your name", placeholder="Enter your name here...", value=st.session_state.get("name", ""))

st.title("🤩 Find your friends APP 🤩")


if st.session_state["name"]:
    st.header(f"""Hello, {st.session_state["name"]}!
Find persons which have similar personalities.""")

    @st.cache_data
    def get_model():
        return load_model(MODEL_NAME)
    
    @st.cache_data
    def get_cluster_names_and_descriptions():
        with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
            return json.loads(f.read())

    @st.cache_data
    def get_all_participants():
        model = get_model()
        all_df = pd.read_excel(DATA)
        df_with_clusters = predict_model(model, data=all_df)
        return df_with_clusters

    with st.sidebar:

        age = st.selectbox("Select your age", ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", ">=65", 'unknown'])
        edu_level = st.radio("Select your education", ["primary", "secondary", "higher"])
        fav_animals = st.selectbox("Select your favorite animals", ["cats", "dogs", "exotic", "cats & dogs", "No favourites"])
        gender = st.radio("Select your gender", ["Woman", "Man"])
        fav_place = st.selectbox("Select your favorite places", ["By the water", "In the forest", "In the mountains", "Other"])

        person_df = pd.DataFrame([
            {
                'age': age,
                'edu_level': edu_level,
                'fav_animals': fav_animals,
                'fav_place': fav_place,
                'gender': gender
            }
        ])

    model = get_model()
    all_df = get_all_participants()
    cluster_names_and_descriptions = get_cluster_names_and_descriptions()


    st.write("Your data:")
    st.dataframe(person_df, hide_index=True)

    predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
    predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

    st.header(f"The closest Cluster to you is {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])   
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.metric("Number of your friends", len(same_cluster_df))

    st.header("Persons in the group")
    fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
    fig.update_layout(
        title="Age distribution in the group",
        xaxis_title="Age",
        yaxis_title="Number of people",
    )
    st.plotly_chart(fig)

    fig_pie_gender = px.pie(
        same_cluster_df,
        names="gender",
        title="Gender Distribution in the Group",
        hole=0.3,
        color_discrete_sequence=["#636EFA", "#EF553B"]
    )
    st.plotly_chart(fig_pie_gender)

    fig_pie_edu = px.pie(
        same_cluster_df,
        names="edu_level",
        title="Education Level Distribution in the Group",
        hole=0.3,
        color_discrete_sequence=["#00CC96", "#AB63FA", "#FFA15A"]
    )
    st.plotly_chart(fig_pie_edu)

    fig_density = px.density_heatmap(
        same_cluster_df,
        x="age",
        y="fav_place",
        z=None,
        color_continuous_scale="Viridis",
        title="Heatmap: Age vs Favorite Place",
        labels={"fav_place": "Favorite Place"},
    )
    st.plotly_chart(fig_density)

    fig_box_age = px.box(
        all_df,
        x="Cluster",
        y="age",
        color="Cluster",
        title="Age Distribution Across Clusters",
        labels={"age": "Age", "Cluster": "Cluster ID"}
    )
    st.plotly_chart(fig_box_age)

    fav_animals_dist = all_df.groupby(['Cluster', 'fav_animals']).size().reset_index(name='count')
    fig_bar_animals = px.bar(
        fav_animals_dist,
        x="fav_animals",
        y="count",
        color="Cluster",
        barmode="group",
        title="Favorite Animals Distribution Across Clusters"
    )
    st.plotly_chart(fig_bar_animals)

else:
    st.info("Please enter your name to start.")
