# stages/stage_3/likelihoods.py

def demand_spike_likelihood(context):
    deviations = context["relational"]["feature_deviations"]
    directions = [v["direction"] for v in deviations.values()]
    z_scores = [abs(v["z_score"]) for v in deviations.values()]

    if all(d == "up" for d in directions) and sum(z_scores) / len(z_scores) > 1.5:
        return 0.8
    elif directions.count("up") >= 2:
        return 0.4
    else:
        return 0.1


def supply_disruption_likelihood(context):
    deviations = context["relational"]["feature_deviations"]
    directions = [v["direction"] for v in deviations.values()]

    if "up" in directions and "down" in directions:
        return 0.8
    elif directions.count("up") == 1 or directions.count("down") == 1:
        return 0.5
    else:
        return 0.1


def data_quality_likelihood(context):
    deviations = context["relational"]["feature_deviations"]
    z_scores = sorted([abs(v["z_score"]) for v in deviations.values()], reverse=True)

    if z_scores[0] > 4 and z_scores[1] < 1:
        return 0.85
    elif z_scores[0] > 3:
        return 0.5
    else:
        return 0.1
