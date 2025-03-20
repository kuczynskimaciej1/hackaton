import pandas as pd

transformation = {
            'CDMA': 0,
            'GSM': 1,
            'LTE': 2,
            'NR': 3,
            'UMTS': 4
        }

def finish_with_prediciton(model):
    prod_data = pd.read_parquet('production_data_preprocessed.parquet')
    predicted = model.predict(prod_data)
    predicted_df = pd.DataFrame({'radio': predicted})

    def rev_radio(x):
        # return key from transformation dict if value is x
        for key, value in transformation.items():
            if value == x:
                return key

    predicted_df['radio'] = predicted_df['radio'].apply(rev_radio)

    predicted_df.to_parquet('solution.parquet')
    print(predicted_df.head())
