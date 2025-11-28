

# Dictionary of location-specific data
locations_data = {
    'Houston': {
        'region': 'COAST',
        'weather_columns': ["Temperature", "Humidity", "Wind", "Pressure", "Precipitation", "Condition"],
        'n_households': 2428417,
        'demand_downscale_rate': 0.1005,
        'weather_url': "https://www.wunderground.com/history/daily/us/tx/houston/KHOU/date/"
    },
    'Austin': {
        'region': 'SCENT',
        'weather_columns': ["Temperature", "Humidity", "Wind", "Pressure", "Precipitation", "Condition"],
        'n_households': 2002462,
        'demand_downscale_rate': 0.1005,
        'weather_url': "https://www.wunderground.com/history/daily/us/tx/austin/KGTU/date/"
    }
}


def pick_location(location):
    return locations_data[location]
