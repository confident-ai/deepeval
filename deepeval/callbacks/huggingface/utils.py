def get_column_order(scores: dict):
    order = ["epoch", "step", "loss", "learning_rate"]
    order.extend([key for key in scores.keys() if key not in order])
    
    return order