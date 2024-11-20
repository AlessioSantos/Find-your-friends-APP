import json
import streamlit as st
import pandas as pd
from pycaret.clustering import load_model, predict_model
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch  

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
    predicted_cluster_data = cluster_names_and_descriptions[str(predicted_cluster_id)]

    st.header(f"The closest Cluster to you is {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])
    same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
    st.metric("Number of your friends", len(same_cluster_df))

    st.header("Persons in the group")

    # Существующие графики
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

   
    st.header("Venn Diagram of Attributes in Your Cluster")

    
    if len(same_cluster_df) >= 2:
        
        set_animals = set(same_cluster_df[same_cluster_df['fav_animals'] == 'dogs'].index)
        set_place = set(same_cluster_df[same_cluster_df['fav_place'] == 'By the water'].index)
        set_edu = set(same_cluster_df[same_cluster_df['edu_level'] == 'higher'].index)

        
        plt.figure(figsize=(8, 8))
        venn = venn3(
            [set_animals, set_place, set_edu],
            set_labels=('Dog Lovers', 'Prefer Water', 'Higher Education')
        )

        
        subset_ids = ('100', '010', '001', '110', '101', '011', '111')
        colors = {
            '100': '#FF9999',  
            '010': '#66B2FF',  
            '001': '#99FF99',  
            '110': '#FFCC99',  
            '101': '#C2C2F0',  
            '011': '#F0E68C',  
            '111': '#D3D3D3'   
        }

        for subset_id in subset_ids:
            patch = venn.get_patch_by_id(subset_id)
            if patch:
                patch.set_color(colors[subset_id])
                patch.set_alpha(0.7)

        
        plt.title('Overlap Between Dog Lovers, Water Preference, and Higher Education in Your Cluster')

        
        st.pyplot(plt.gcf())
    else:
        st.write("Not enough data in this cluster to display the Venn Diagram.")

    
    st.header("3D Spiral Histogram of Participant Attributes in Your Cluster")

    
    if len(same_cluster_df) >= 2:
        
        features = ['age', 'edu_level', 'fav_animals', 'fav_place', 'gender']

        
        data_numeric = same_cluster_df.copy()
        for col in features:
            data_numeric[col] = data_numeric[col].astype('category').cat.codes

        
        theta = np.linspace(0, 4 * np.pi, len(data_numeric))
        z = np.linspace(-2, 2, len(data_numeric))
        r = np.linspace(0.1, 1, len(data_numeric))

        
        fig_spiral = plt.figure(figsize=(14, 10))
        ax = fig_spiral.add_subplot(111, projection='3d')

        
        colors = plt.cm.plasma(np.linspace(0, 1, len(features)))

        
        for index, column in enumerate(features):
            values = data_numeric[column].values
            x = r * np.sin(theta + index)
            y = r * np.cos(theta + index)
            dz = values

            
            ax.bar3d(x, y, z, dx=0.05, dy=0.05, dz=dz, color=colors[index], alpha=0.7)

            
            z += 0.4

        
        ax.set_title('3D Spiral Histogram of Participant Attributes in Your Cluster', fontsize=20)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Attribute Values')

        
        ax.view_init(elev=30, azim=120)

        
        legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=features[i]) for i in range(len(features))]
        ax.legend(handles=legend_elements, loc='upper right')

        
        st.pyplot(fig_spiral)
    else:
        st.write("Not enough data in this cluster to display the 3D Spiral Histogram.")

else:
    st.info("Please enter your name to start.")
