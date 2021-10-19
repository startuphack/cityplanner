import json
from operator import itemgetter

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events


@st.cache()
def load_data():
    with open("resources/adm_names.json") as f:
        adm_names = json.load(f)

    school_types = [
        {"name": "predefined1", "n_pupils": 2000},
        {"name": "predefined2", "n_pupils": 2500},
    ]
    return dict(
        adm_names=adm_names,
        school_types=school_types,
    )


data = load_data()

st.title("Планирование школ")
st.subheader("Subtitle")


st.session_state.setdefault("school_types", data["school_types"].copy())
st.session_state.setdefault("school_types_value", data["school_types"].copy())

param1 = st.sidebar.slider("param1", 0, 10, 5)

st.sidebar.selectbox("Район", data["adm_names"], format_func=data["adm_names"].get, key="region_value")

st.sidebar.multiselect(
    "Типы школ",
    st.session_state["school_types"],
    default=st.session_state.get("school_types_value", []),
    format_func=itemgetter("name"),
)

with st.sidebar.expander("Добавить проект школы", expanded=False):
    with st.form("new_school_type"):
        st.text_input("Название", key="new_school_name")
        st.number_input(
            "Кол-во учеников",
            min_value=100,
            max_value=100_000,
            value=2000,
            step=100,
            key="new_school_n_pupils",
        )
        checkbox_val = st.checkbox("Form checkbox")

        def on_submit():
            if not st.session_state["new_school_name"]:
                st.sidebar.error("Название не может быть пустым")
                return

            if st.session_state["new_school_name"] in (
                item["name"] for item in st.session_state["school_types"]
            ):
                st.sidebar.error("Проект с таким названием уже есть")
                return

            st.session_state["school_types"].append(
                {
                    "name": st.session_state["new_school_name"],
                    "n_students": st.session_state["new_school_n_pupils"],
                }
            )
            st.sidebar.success("Новый тип успешно добавлен")

        st.form_submit_button("Добавить", on_click=on_submit)


print("Region =", st.session_state["region_value"])
print("School types =", st.session_state["school_types"])

col1, col2 = st.columns(2)
with col1:
    fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
    selected_points = plotly_events(fig)
    print("selectd points =", selected_points)

with col2:
    map_data = [
        [55.779698316547616, 37.68008274325365, 0.07588035190615836],
        [55.785951580391234, 37.488155312444505, 0.11135718475073314],
        [55.78500267320341, 37.59571056696728, 0.0821208211143695],
        [55.784492269012034, 37.64948577621047, 0.17214310850439882],
        [55.78441739849473, 37.657167813282726, 0.11763049853372434],
    ]
    st.subheader("Map")
    st.map(pd.DataFrame.from_records(map_data, columns=["lat", "lon", "val"]))
