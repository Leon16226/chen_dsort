
def amend_sim(sim):
    if sim <= 0:
        return 0.05
    elif sim >= 1:
        return 0.99
    else:
        return sim