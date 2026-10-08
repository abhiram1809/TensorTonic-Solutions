def target_encoding(categories: list, targets: list) -> list:
    """
    Returns each category replaced by its mean target.
    """
    build_dict = {}
    for val,tar in zip (categories,targets):
        if val in build_dict:
            build_dict[val][0]+=1
            build_dict[val][1]+=tar
        else:
            build_dict[val] = [1, tar]
    means = {key:(val[1]/val[0]) for key, val in build_dict.items()}
        
    return [means[key] for key in categories]