import json
import time
import re
import pathlib
import subprocess

import pydeck as pdk
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit import session_state as state
from streamlit_plotly_events import plotly_events
from planner.utils.files import pickle_load

ICON_URL = "https://img.icons8.com/plasticine/100/000000/marker.png"

st.set_page_config(layout="wide")
st.title("Планирование школ")


state.setdefault("optimizer_process", None)

@st.cache()
def load_data():
    with open("resources/adm_names.json") as f:
        adm_names = json.load(f)

    return dict(adm_names=adm_names)


data = load_data()


st.sidebar.selectbox("Район", data["adm_names"], format_func=data["adm_names"].get, key="region_value")
uploaded_file = st.sidebar.file_uploader("Загрузить файл конфигурации", type=['xlsx'])
config_path = pathlib.Path('config.xlsx')
if uploaded_file is not None:
    if not config_path.exists() or config_path.stat().st_size != uploaded_file.size:
        with config_path.open('wb') as f:
            f.write(uploaded_file.getvalue())


is_started = state["optimizer_process"] is not None
placeholder = st.sidebar.empty()
results_path = f"results/{state['region_value']}/"

if placeholder.button("Запустить" if not is_started else "Остановить"):
    if is_started:
        state["optimizer_process"].terminate()
        state["optimizer_process"] = None
        placeholder.button("Запустить")
        is_started = False
    else:
        log_file = pathlib.Path('optimization.log').open('a')
        cmd = [
            "python",
            "planner/optimization/run_optimization.py",
            "--adm-id",
            state["region_value"],
            "--results-path",
            results_path,
        ]
        if config_path.exists():
            cmd += ['--config-file', 'config.xlsx']

        state["optimizer_process"] = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

        st.sidebar.success("Оптимизация запущена")
        placeholder.button("Остановить")
        is_started = True


def step_files():
    return list(pathlib.Path(results_path).glob("step_*"))


if not step_files():
    if not is_started:
        st.info("Для просмотра результатов запустите расчет")
        st.stop()

    else:
        with st.spinner("Рассчитываем оптимальные расположения..."):
            while True:
                time.sleep(1)
                if step_files():
                    break

            while True:
                try:
                    optimizer_data = pickle_load(max(step_files()))
                    break
                except EOFError:
                    time.sleep(1)

# берем последний файл step_*
last_step_file = max(step_files(), key=lambda path: int(re.sub(r'\D', '', path.stem)))
n_step = int(last_step_file.name.split('step_')[-1].split('.')[0])
st.subheader(f"Итерация {n_step}")

optimizer_data = pickle_load(last_step_file)
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


icon_data = {
    "url": ICON_URL,
    "width": 128,
    "height": 128,
    "anchorY": 128,
}
map_df["icon_data"] = pd.Series(icon_data for _ in range(map_df.shape[0]))

if map_df.shape[0] > 1:
    view_state = pdk.data_utils.compute_view(map_df[["lon", "lat"]], 1.1)
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

st.pydeck_chart(
    pdk.Deck(
        layers=[icon_layer],
        initial_view_state=view_state,
        tooltip={"text": "Кол-во учеников: {n}"},
        map_style="mapbox://styles/mapbox/streets-v11",
    )
)
