import re


def replace_with_previous(df):
    """Replace 'idem' by the previous cell value."""
    for col in df.columns:
        previous_value = None
        for index, row in df.iterrows():
            value = row[col]
            if value == 'idem' and previous_value is not None:
                df.at[index, col] = previous_value
            else:
                previous_value = value
    return df

def first_preprocessing(df):
    """Preprocessing of the DataFrame for the first model."""
    # Drop rows with all NaN values
    df = df.dropna(how='all')
    
    # Drop les colonnes vides (maiden_name, education_level) + observation car biaise
    df = df.drop(columns=['maiden_name', 'education_level', 'observation'])
    
    # Age
    # If the date is composed of 4 digits, it is not an age, thus replace by NA
    mask = df['age'].notna() & df['age'].astype(str).str.match(r'^\d{4}$')
    df.loc[mask, 'age'] = None

    # Birthdate
    # If the date is not composed of 4 characters, it is not a birth date, thus replace by NA
    mask = df['birth_date'].notna() & ~df['birth_date'].astype(str).str.match(r'^\d{4}$')
    df.loc[mask, 'birth_date'] = None
    
    # Replace "idem" by the previous cell value
    df = replace_with_previous(df)
    
    # Lowercase
    # df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    
    return df

def replace_first_occurrence_chef(match):
        return 'chef de ménage'

def replace_first_occurrence_link(match):
        return 'sa femme'
    
def second_preprocessing(df):
    """Preprocessing of the DataFrame for the second model."""
    # Si l'age ne contient pas semaines, mois, jours, an, ans ou jour et que la longueur est supérieure à 2, alors on remplace par NA
    df['age'] = df['age'].apply(lambda x: None if isinstance(x, float) or (x is not None and not any(word in x for word in ['semaine', 'semaines', 'mois', 'jours', 'an', 'ans', 'jour']) and len(x) > 2) else x)
    
    # impossible de tout modifier
    replace_chef = ['ch d m', 'ch d m.', 'ch d mé', 'ch de m', 'ch de m ge', 'ch de m.', 'ch de ménage', 'ch. d . m', 'ch. d m', 'ch. d m.', 'ch. d. m', 'ch. d. m.', 'ch. de M.', 'ch. de m', 'ch. de m ge', 
                    'ch. de m.', 'ch. de mge', 'ch. de ménage', 'ch. m', 'ch.de ménage', 'ch.m', 'ch.m.', 'chef d m', 'chef de f', 'chef de f le', 'chef de fam', 'chef de famille', 'chef de flle', 
                    'chef de m e', 'chef de m ge', 'chef de m.', 'chef de maison', 'chef de mange', 'chef de mge', 'chef de mé age', 'chef de méange', 'chef de mén.', 
                    'chef de ménage et veuve', 'chef de ménage, son fils', 'chef de m⁇nage', 'chef domestique', 'chef du ménage', 'chef m ge', 'chef ménage', 'CHEF DE MENAGE', 
                    'Ch. de M.', 'Chef de menage', 'Chef de mén', 'Chef de ménage', 'chef de ménage d m', 'chef de ménage de f', 'chef de ménage de f le', 'chef de ménage de fam', 'chef de ménage de famille', 
                    'chef de ménage de flle', 'chef de ménage de menage', 'chef de ménage de mén', 'chef de ménage de ménage', 'chef de ménage de poste', 'chef de ménage domestique', 'chef de ménage du ménage',
                    'chef de ménage ge', 'chef de ménage m ge', 'chef de ménage ménage', 'chef de ménage religieuse', 'chef de ménage veuf', 'chef de ménage.', 'cheg','chf']
    pattern = r'\b(?:' + '|'.join(map(re.escape, replace_chef)) + r')\b'
    df['link'] = df['link'].str.replace(pattern, replace_first_occurrence_chef, 1, regex=True)

    replace_link = ['s f e', 'sa f', 'sa f e', 'sa f me', 'sa fe', 'sa felle', 'sa fem', 'sa femlme']
    pattern2 = r'\b(?:' + '|'.join(map(re.escape, replace_link)) + r')\b'
    df['link'] = df['link'].str.replace(pattern2, replace_first_occurrence_link, 1, regex=True)
    
    return df