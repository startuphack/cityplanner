import os
import json
import time
import re
import pathlib
import itertools
import subprocess

import pydeck as pdk
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.neighbors import BallTree
from streamlit import session_state as state
from streamlit_plotly_events import plotly_events
from planner.utils.files import pickle_load
from planner.optimization import loaders
from annotated_text import annotated_text, annotation

EARTH_SIZE = 6371  # km
ICON_URL = "https://img.icons8.com/plasticine/100/000000/marker.png"

st.set_page_config(layout="wide")
st.title("Планирование школ")

state.setdefault("optimizer_process", None)


def load_data():
    with open("resources/adm_names.json") as f:
        adm_names = json.load(f)

    files_df = pd.DataFrame({'files': pathlib.Path('results').glob('*/*')})

    files_df['mtime'] = files_df.files.apply(lambda x: os.path.getmtime(str(x)))
    files_df['parent'] = files_df.files.apply(lambda x: str(x.parent.name))
    mtimes_index = files_df.groupby('parent').agg({'mtime': max}).to_dict()['mtime']

    sorted_names = sorted(adm_names.items(), key=lambda x: mtimes_index.get(x[0], 0), reverse=True)
    adm_names = dict(sorted_names)

    schools = loaders.load_schools()

    schools_idx = BallTree(pd.DataFrame({'x': np.radians(schools.geometry.x), 'y': np.radians(schools.geometry.y)}),
                           metric='haversine')

    return dict(
        adm_names=adm_names,
        shapes=loaders.load_shapes(),
        adm_zones=loaders.load_adm_zones(),
        schools_idx=schools_idx,
    )


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

        placeholder.button("Остановить")
        is_started = True

if is_started:
    return_code = state["optimizer_process"].poll()
    if return_code is None:
        st.sidebar.success("Оптимизация запущена")
    else:
        state["optimizer_process"] = None
        if return_code == 0:
            st.sidebar.success("Оптимизация завершена")
        elif return_code == 101:
            st.sidebar.info("Нет учеников для размещения")
        else:
            st.sidebar.error("Произошла ошибка")


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

hover_list = list()
hover_columns = [
    'число объектов',
    'общая стоимость реализации, млрд',
    'среднее расстояние до школы для ученика, км',
    'удобство, %',
    'необходимо мест',
    'запланировано мест',
    'перцентиль(50) расстояния до объекта',
    'перцентиль(70) расстояния до объекта',
    'перцентиль(90) расстояния до объекта',
    'перцентиль(95) расстояния до объекта',
]

rename_columns = {
    "стоимость, млрд": "общая стоимость реализации, млрд",
    "среднее расстояние, км": "среднее расстояние до школы для ученика, км",
    "необходимо метст": "необходимо мест",
}

plot_df.rename(columns=rename_columns, inplace=True)
plot_df['obj-num'] = plot_df['число объектов'].apply(lambda x: f'число школ в проекте = {x}')


fig = px.scatter(
    plot_df,
    x="общая стоимость реализации, млрд",
    y="среднее расстояние до школы для ученика, км",
    color="удобство, %",
    hover_name='obj-num',
    hover_data={
        'общая стоимость реализации, млрд': True,
        'среднее расстояние до школы для ученика, км': True,
        'size': False,
        'удобство, %': True,
        'необходимо мест': True,
        'запланировано мест': True,
        'перцентиль(50) расстояния до объекта': True,
        'перцентиль(70) расстояния до объекта': True,
        'перцентиль(90) расстояния до объекта': True,
        'перцентиль(95) расстояния до объекта': True,
    },
    size="size",
    # color_continuous_scale="RdBu",
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
else:
    objects = []

shapes = data['shapes'][data['shapes'].adm_zid == (int(state["region_value"]))]
adm_zone = data['adm_zones'][data['adm_zones'].adm_zid == int(state['region_value'])]

icon_data = {
    "url": ICON_URL,
    "width": 128,
    "height": 128,
    "anchorY": 128,
}
map_df["icon_data"] = pd.Series(icon_data for _ in range(map_df.shape[0]))

icon_layer = pdk.Layer(
    type="IconLayer",
    data=map_df,
    get_icon="icon_data",
    get_size=4,
    size_scale=15,
    get_position=["lon", "lat"],
    pickable=True,
)

borders_layer = pdk.Layer(
    "GeoJsonLayer",
    adm_zone,
    stroked=True,
    filled=False,
    get_line_width=20,
    get_line_color=[189, 0, 38],
)

# находим расстояние до ближайшей школы для каждого квадрата
shapes_df = pd.DataFrame({'x': np.radians(shapes.geometry.centroid.x), 'y': np.radians(shapes.geometry.centroid.y)})
dist, _ = data['schools_idx'].query(shapes_df)
dist *= EARTH_SIZE

grid_layer = pdk.Layer(
    "GridLayer",
    pd.DataFrame({
        'coords': shapes.geometry.centroid.map(lambda p: [p.x, p.y]),
        'weight': dist.ravel() * shapes['customers_cnt_home'],
    }),
    opacity=0.1,
    cell_size=500,
    get_position="coords",
    get_color_weight="weight",
)

view_state = pdk.data_utils.compute_view(adm_zone.geometry.convex_hull.exterior.iloc[0].coords, 1.1)

st.pydeck_chart(
    pdk.Deck(
        layers=[borders_layer, grid_layer, icon_layer],
        initial_view_state=view_state,
        tooltip={"text": "Кол-во учеников: {n}"},
        map_style="mapbox://styles/mapbox/streets-v11",
    )
)

levels = (dist.ravel() * shapes['customers_cnt_home'] / 1000).quantile(np.linspace(0, 1, 6)).to_list()
st.caption('Суммарное расстояние до школы для учеников сектора на 1000 человек')
annotated_text(
    annotation(f'< {levels[1]:.1f}', background='#ffffb2'),
    annotation(f'< {levels[2]:.1f}', background='#fed976'),
    annotation(f'< {levels[3]:.1f}', background='#feb24c'),
    annotation(f'< {levels[4]:.1f}', background='#fd8d3c'),
    annotation(f'< {levels[5]:.1f}', background='#f03b20'),
    annotation(f'> {levels[5]:.1f}', background='#bd0026'),
)
if objects:
    st.subheader('Проекты')
    st.dataframe(pd.DataFrame({'Проект': [o.project['name'] for o in objects], 'Кол-во': 1}).groupby('Проект').count())
