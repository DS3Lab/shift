def process_other(candidates, other):
    original = len(candidates)
    excludes = other["value"]["models"]
    for each in excludes:
        if "num_params" in each:
            del each["num_params"]
    for each in candidates[:]:
        if each in excludes:
            candidates.remove(each)
    new = len(candidates)
    print("[INFO] {} candidate models reduced to {}".format(original, new))
    return candidates
