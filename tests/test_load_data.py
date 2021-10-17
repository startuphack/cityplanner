import planner.optimization.loaders as L


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
