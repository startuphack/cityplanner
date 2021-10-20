import json
import time
import pathlib
import subprocess

import pydeck as pdk
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state
from streamlit_plotly_events import plotly_events
from planner.utils.files import pickle_load

st.set_page_config(layout="wide")


@st.cache()
def load_data():
    with open("resources/adm_names.json") as f:
        adm_names = json.load(f)

    return dict(adm_names=adm_names)


data = load_data()

st.title("Планирование школ")
st.subheader("Subtitle")

state.setdefault("optimizer_process", None)

st.sidebar.selectbox("Район", data["adm_names"], format_func=data["adm_names"].get, key="region_value")

is_started = state["optimizer_process"] is not None
placeholder = st.sidebar.empty()
if placeholder.button("Запустить" if not is_started else "Остановить"):
    if is_started:
        state["optimizer_process"].terminate()
        state["optimizer_process"] = None
        placeholder.button("Запустить")
        is_started = False
    else:
        state["optimizer_process"] = subprocess.Popen(["sleep", "60"])  # todo
        st.sidebar.success("Оптимизация запущена")
        placeholder.button("Остановить")
        is_started = True


def step_files():
    return list(pathlib.Path("resources/opt/").glob("step_*"))


if not step_files():
    if not is_started:
        st.info("Для просмотра результатов запустите расчет")
        st.stop()

    else:
        with st.spinner("Рассчитываем оптимальные расположения..."):
            while True:
                time.sleep(0.5)
                if step_files():
                    break


# берем последний файл step_*
optimizer_data = pickle_load(max(step_files()))
plot_df = optimizer_data["plot_df"]
factory = optimizer_data["factory"]
point_col = plot_df["point"]
del plot_df["point"]

fig = px.scatter(
    plot_df,
    x="стоимость, млрд",
    y="среднее расстояние",
    color="неудобство",
    hover_data=plot_df.columns,
    size="size",
    color_continuous_scale="RdBu",
)
fig.update_layout(
    plot_bgcolor="#F0F2F6",
    hoverlabel_font={"family": "Courier New", "size": 16},
)

map_df = pd.DataFrame(columns=["lat", "lon"])
selected_points = plotly_events(fig, override_width="100%", override_height=550)
if len(selected_points) == 1:
    idx = selected_points[0]["pointIndex"]
    objects = factory.make_objects_from_point(point_col.iloc[idx])
    map_df = pd.DataFrame.from_records(
        [[*obj.coords(), obj.num_peoples] for obj in objects],
        columns=["lat", "lon", "n"],
    )




ICON_URL = "https://img.icons8.com/plasticine/100/000000/marker.png"

icon_data = {
    "url": ICON_URL,
    "width": 128,
    "height": 128,
    "anchorY": 128,
}

map_df["icon_data"] = None
for i in map_df.index:
    map_df["icon_data"][i] = icon_data

if map_df.shape[0] > 0:
    view_state = pdk.data_utils.compute_view(map_df[["lon", "lat"]], 0.6)
else:
    view_state = pdk.ViewState(longitude=37.618423, latitude=55.751244, zoom=10, pitch=0)

icon_layer = pdk.Layer(
    type="IconLayer",
    data=map_df,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position=["lon", "lat"],
    pickable=True,
)

r = pdk.Deck(
    layers=[icon_layer],
    initial_view_state=view_state,
    tooltip={"text": "Кол-во учеников: {n}"},
    map_style="mapbox://styles/mapbox/streets-v11",
)
st.pydeck_chart(r)
