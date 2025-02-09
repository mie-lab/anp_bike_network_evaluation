ROAD_WIDTH_MAPPING = {
        "4m Strasse": 4,
        "3m Strasse": 3,
        "6m Strasse": 6,
        "10m Strasse": 10,
        "8m Strasse": 8,
        "Autobahn": 25,
        "Verbindung": 6,
        "Ausfahrt": 5,
        "Einfahrt": 5,
        "Dienstzufahrt": 3,
        "Autostrasse": 10,
        "Zufahrt": 3
    }

NOISE_POLLUTION_MAPPING = {
    "lre_tag": "day_noise_emissions",
    "vt": "speed_day",
    "nt1": "light_veh_volume",
    "nt2": "heavy_veh_volume",
    "anz_tram_tag": "tram_daily_volume",
    "strassentyp": "road_type"
    }

ROAD_TYPE_MAPPING = {
      "HLS": "Highway",
      "VS": "MajorRoad",
      "SS": "CollectorRoad"
    }

TRAFFIC_VOLUME_MAPPING = {
    "DTV_FZG": "AADT_all_veh",
    "DTV_PW": "AADT_personal_veh",
    "DTV_LI": "AADT_delivery_veh",
    "DTV_LW": "AADT_truck_veh",
    "DTV_LZ": "AADT_articulated_truck_veh"
    }

ZONE_MAPPING = {
        "W2": "residential",
        "W2bI": "residential",
        "W2bII": "residential",
        "W2bIII": "residential",
        "W3": "residential",
        "W4": "residential",
        "W5": "residential",
        "Z5": "commercial",
        "Z6": "commercial",
        "Z7": "commercial",
        "IHD": "industrial",
        "I": "industrial zone",
        "Oe": "public",
        "Oe2": "public",
        "Oe2F": "public",
        "Oe3": "public",
        "Oe3F": "public",
        "Oe4": "public",
        "Oe4F": "public",
        "Oe5": "public",
        "Oe5F": "public",
        "Oe6": "public",
        "Oe7": "public",
        "OeI": "public",
        "OeII": "public",
        "OeIII": "public",
        "QI": "conservation",
        "QII": "conservation",
        "QIII": "conservation",
        "K": "core",
        "E1": "recreation_ground",
        "E2": "recreation_ground",
        "E3": "recreation_ground",
        "F": "clearance_zone",
        "FA": "common_land",
        "FC": "sports_swimming_facilities",
        "FD": "campsite",
        "FE": "cemetery",
        "Fk": "cantonal/regional_clearance_zone",
        "L": "agricultural",
        "R": "reserve_zone",
        "(Forest)": "forest",
        "(Water bodies)": "basin"
    }

SURFACE_MAPPING = {
        "AC 11": 5,
        "AC 8": 5,
        "SMA 11": 5,
        "AC MR 8": 5,
        "unbekannt diverse": 1,
        "Pflästerung unbekannt": 1,
        "AC 16": 5,
        "OB": 1,
        "SMA 8": 5,
        "Pflästerung Sandfuge": 2,
        "SDA 8": 5,
        "GA 11": 1,
        "Pflästerung Mörtelfuge": 2,
        "AC 6": 5,
        "Chaussierung": 3,
        "SDA 4": 5,
        "AC 22": 5,
        "SMA 16": 5,
        "AC 8 lärmarm": 5,
        "HRA": 1,
        "AC MR 11": 5
      }
