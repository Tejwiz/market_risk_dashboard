import pandas as pd

def apply_stress_scenario(df, scenario):
    df = df.copy()
    if scenario == "2008 Crisis":
        df['Shock %'] = -0.40
    elif scenario == "COVID Crash":
        df['Shock %'] = -0.25
    else:  # Custom scenario
        df['Shock %'] = -0.10  # default -10% shock

    df['Change'] = df['Price'] * df['Quantity'] * df['Shock %']
    return df
