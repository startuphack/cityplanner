import json
import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from planner.optimization import loaders
from planner.utils.files import resources


def load_data():
    with open(f"{resources}/adm_names.json") as f:
        adm_names = json.load(f)

    files_df = pd.DataFrame({'files': pathlib.Path('results').glob('*/*')})

    files_df['mtime'] = files_df.files.apply(lambda x: os.path.getmtime(str(x)))
    files_df['parent'] = files_df.files.apply(lambda x: str(x.parent.name))
    mtimes_index = files_df.groupby('parent').agg({'mtime': max}).to_dict()['mtime']

    sorted_names = sorted(adm_names.items(), key=lambda x: mtimes_index.get(x[0], 0), reverse=True)
    adm_names = dict(sorted_names)

    schools = loaders.load_schools()
    schools_idx = BallTree(pd.DataFrame({'x': np.radians(schools.geometry.x), 'y': np.radians(schools.geometry.y)}), metric='haversine')

    return dict(
        adm_names=adm_names,
        shapes=loaders.load_shapes(),
        adm_zones=loaders.load_adm_zones(),
        schools_idx=schools_idx,
    )





if __name__ == '__main__':
    data = load_data()
    adm_id=114
    shapes = data['shapes'][data['shapes'].adm_zid == adm_id]
    adm_zone = data['adm_zones'][data['adm_zones'].adm_zid == adm_id]

    shapes_df = pd.DataFrame({'x': np.radians(shapes.geometry.centroid.x), 'y': np.radians(shapes.geometry.centroid.y)})
    dist, _ = data['schools_idx'].query(shapes_df)
    res_df = pd.DataFrame({
        'coords': shapes.geometry.centroid.map(lambda p: [p.x, p.y]),
        'weight': dist.ravel() * shapes['customers_cnt_home'],
    })

    print(res_df)
