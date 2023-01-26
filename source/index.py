from statistic_training_pipeline import train_stats_model


def handler(event, context):
    return train_stats_model(event)
