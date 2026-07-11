def child_feature(df):
    df['Child'] = df['Age'] < 10
    df['Child'] = df['Child'].astype(int)

def family_size_feature(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1