def pool_qc(df):
    df['PoolQC'] = df['PoolQC'].notna()
    return df

def misc_feature(df):
    df['MiscFeature'] = df['MiscFeature'].notna()
    return df

def alley(df):
    df['Alley'] = df['Alley'].fillna('NoAlley')
    return df

def fence(df):
    df['Fence'] = df['Fence'].fillna('NoFence')
    return df

def mas_vnr_type(df):
    df['MasVnrType'] = df['MasVnrType'].fillna('NoMasonryVeneer')
    return df

def fireplace_qu(df):
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None').map(quality_map)
    return df

def lot_frontage(df):
    df['LotFrontage'] = df['LotFrontage'].fillna(df.groupby('Neighborhood')['LotFrontage'].transform('median'))

def garage_qual(df):
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df['GarageQual'] = df['GarageQual'].fillna('None').map(quality_map)
    return df

def garage_type(df):
    df['GarageType'] = df['GarageType'].fillna('NoGarage')
    return df

def garage_finish(df):
    df['GarageFinish'] = df['GarageFinish'].fillna('None')
    return df

def garage_cond(df):
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    df['GarageCond'] = df['GarageCond'].fillna('None').map(quality_map)
    return df

def bsmt_cond(df):
    df['BsmtCond'] = df['BsmtCond'].fillna('None')
    return df

def bsmt_qual(df):
    df['BsmtQual'] = df['BsmtQual'].fillna('None')
    return df

def bsmt_exposure(df):
    quality_map = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
    df['BsmtExposure'] = df['BsmtExposure'].fillna('None').map(quality_map)
    return df

def bsmt_fin_type1(df):
    quality_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None').map(quality_map)
    return df

def bsmt_fin_type2(df):
    quality_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None').map(quality_map)
    return df

def garage_yr_blt(df):
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt'])
    return df

def mas_vnr_area(df):
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    return df

def electrical(df):
    df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
    return df

def delete_outliers(df):
    df = df.drop([523, 1298])
    return df

def clip_rare_values(df):
    df['GarageCars'] = df['GarageCars'].clip(upper=3)
    df['Fireplaces'] = df['Fireplaces'].clip(upper=2)
    df['TotRmsAbvGrd'] = df['TotRmsAbvGrd'].clip(lower=3, upper=11)
    return df