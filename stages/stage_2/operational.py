# stages/stage_2/operational.py

def extract_operational_context(config):
    return {
        "system": config.get("operational", {}).get("system"),
        "region": config.get("operational", {}).get("region"),
        "owner_team": config.get("operational", {}).get("owner_team")
    }
