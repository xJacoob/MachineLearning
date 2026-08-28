def cryosleep(df):
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['MoneySpend'] = df[spend_cols].sum(axis=1)

    not_sleep = df['CryoSleep'].isna() & (df['MoneySpend'] > 0)
    df.loc[not_sleep, 'CryoSleep'] = False

    sleep = df['CryoSleep'].isna() & (df['MoneySpend'] == 0)
    df.loc[sleep, 'CryoSleep'] = True

    df['CryoSleep'] = df['CryoSleep'].astype(int)

    return df

def money_columns(df, medians=None):
    spend_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    if medians is None:
        medians = {col: df.loc[df['CryoSleep'] == False, col].median() for col in spend_cols}

    for col in spend_cols:
        not_sleeping =(df['CryoSleep'] == False) & (df[col].isna())
        df.loc[not_sleeping, col] = medians[col]

        sleeping = (df['CryoSleep'] == True) & (df[col].isna())
        df.loc[sleeping, col] = 0.0

    return df, medians

def cabin(df):
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    return df

def vip(df):
    df['VIP'] = df['VIP'].map({True: 1, False: 0})
    return df

def passenger(df):
    df['Group'] = df['PassengerId'].str[:4].astype(int)
    return df


