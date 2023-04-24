from dt import RandomlyInitializedEpsGreedyLeaf

def string_to_dict(x):
    """
    This function splits a string into a dict.
    The string must be in the format: key0-value0#key1-value1#...#keyn-valuen
    """
    result = {}
    items = x.split("#")

    for i in items:
        key, value = i.split("-")
        try:
            result[key] = int(value)
        except:
            try:
                result[key] = float(value)
            except:
                result[key] = value

    return result

