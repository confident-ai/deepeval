def reorder(dic):
    order = ["epoch", "step", "loss", "learning_rate"]
    order.extend([key for key in dic.keys() if key not in order])
    
    return order