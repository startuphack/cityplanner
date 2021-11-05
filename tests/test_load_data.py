import planner.optimization.loaders as L
from planner.utils.geo_utils import create_limits_df
from planner.utils.files import resources


def test_load_schools():
    schools = L.load_schools()
    assert len(schools) > 0


def test_load_eco():
    eco_data = L.load_eco_data()
    assert len(eco_data) > 0


def test_load_buildings():
    buildings = L.load_buildings_data()
    assert len(buildings) > 0


def test_load_highways():
    highways = L.load_highways()
    assert len(highways) > 0


def test_load_water():
    waters = L.load_water()
    assert len(waters) > 0


def test_load_all_eco_data():
    all_eco_data = L.load_all_eco_data()

    assert len(all_eco_data) > 0



def test_load_osm_data():
    create_limits_df('Moscow', tags = {'cemetery':True}, obj_type='cemetery')
    assert (resources/'cemetery.gz.pq').exists()
