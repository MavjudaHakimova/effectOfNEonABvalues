from sqlalchemy import text
import pandas as pd
from sqlalchemy import create_engine

def load_to_postgis(df, df_name):
    print(df)
    engine = create_engine('postgresql://gis:123456@10.0.62.59:55432/gis')

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS earthquakes_GR_low.test_table (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        value FLOAT
    );
    """
    try:
        with engine.connect() as conn:
            conn.execute(text(create_table_sql))
            conn.commit()

        df.to_postgis(
            df_name,
            engine,
            schema='earthquakes_GR_low',
            if_exists='append',
            index=False,
        )
        print(f"[{df_name}] Records created successfully")
    except Exception as ex:
        print("Operation failed: {0}".format(ex))


data = {
    "name": ["point1", "point2"],
    "value": [10.5, 20.7]
}

df = pd.DataFrame(data)
df_name = "test_table"

load_to_postgis(df, df_name)