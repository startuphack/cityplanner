from planner.utils import files


def test_load_configs():
    config_data = files.parse_excel_config(files.resources / 'excel_config.xlsx')

    assert len(config_data['projects']) > 0
    assert len(config_data['limits']) > 0
    assert len(config_data['configs']) > 0
