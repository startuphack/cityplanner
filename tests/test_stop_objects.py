from planner.optimization.model import StopObjects


def test_stop_polygons():
    stop_distances = {
        'beach': 0.3, # не менее 300 метров до пляжа
    }

    stop_objects = StopObjects(stop_distances)
    stop_objects.build_stop_index()
    beach_coords = [
        {
            'lat': 55.79155119614954,
            'lon': 37.41327158681704,
        }
    ]

    for beach_coord in beach_coords:
        assert stop_objects.is_stopped(beach_coord['lat'], beach_coord['lon'])

    far_from_moskow_coords = [
        {
            'lat': 52.79155119614954,
            'lon': 32.41327158681704,
        }
    ]

    for far_coords in far_from_moskow_coords:
        assert not stop_objects.is_stopped(far_coords['lat'], far_coords['lon'])
