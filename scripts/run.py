import pandas as pd
from projectcode import processing, classification

train_data = processing.process_data('data/train.csv')
train_data = train_data.drop('PassengerId', axis=1)

test_data = processing.process_data('data/test.csv')
test_passenger_ids = test_data.PassengerId
test_data = test_data.drop('PassengerId', axis=1)

predictions = classification.run_rforest(train_data.values, test_data.values)

pred_df = classification.write_output(test_passenger_ids, predictions)
