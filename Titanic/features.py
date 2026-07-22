def child_feature(df):
    df['Child'] = df['Age'] < 10
    df['Child'] = df['Child'].astype(int)

def family_size_feature(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

def name_feature(df):
    df['Title'] = df['Name'].str.extract(r',\s*(.*?)\.')

def group_rare_title(df):
    rare_titles = ['Dr', 'Rev', 'Major', 'Mlle', 'Col', 'Don', 'Mme', 'Ms', 'Lady', 'Sir', 'Capt', 'the Countess', 'Jonkheer']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

def has_cabin_features(df):
    df['HasCabin'] = df['Cabin'].notna().astype(int)

def fare_per_person_features(df):
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
