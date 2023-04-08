import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
# check version number
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler

KATEGORIK_SARTI = 10

"""df = pd.read_csv('kk_urun_tahmin.csv', sep=';', encoding='windows-1254')

print(df.dtypes)"""

"""#####Null Analizi##########
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

missing_value_df = missing_value_df[
    (missing_value_df['percent_missing'] >= 60)]  ##Null oranı %60ın üstündeki kolonları sildim.
missing_value_df['column_name'].values"""

# df = df.drop(columns=missing_value_df['column_name'].values,axis=1)
# df.to_csv('modified_kk_eski.csv', index=False, sep=';',encoding='ISO:8859-1')##Yeni csvye yazdım.

df = pd.read_csv(r'modified_kk.csv', sep=';', encoding='windows-1254', low_memory=False)

#####Null Analizi#########
percent_missing = df.isnull().sum() * 100 / len(df)
missing_value_df = pd.DataFrame({'column_name': df.columns,
                                 'percent_missing': percent_missing})

missing_value_df = missing_value_df[
    (missing_value_df['percent_missing'] >= 60)]  ##Null oranı %60ın üstündeki kolonları sildim.
missing_value_df['column_name'].values

"""
#########NULL DOLDURMA############
###MUST_EGITIM_DURUM
a = df['MUST_EGITIM_DURUM'].mode().values  ##Lise en cok tekrar eden
a = '\n'.join(a)
df['MUST_EGITIM_DURUM'] = df['MUST_EGITIM_DURUM'].fillna(a)

####MUST_MESLEK
a = 'BILINMIYOR'
df['MUST_MESLEK'] = df['MUST_MESLEK'].fillna(a)

##MUST_MEDENI_HAL
a = df['MUST_MEDENI_HAL'].mode().values  ##Evli en cok tekrar eden
a = '\n'.join(a)
df['MUST_MEDENI_HAL'] = df['MUST_MEDENI_HAL'].fillna(a)

##MUST_CINSIYET
a = df['MUST_CINSIYET'].mode().values  ##Evli en cok tekrar eden
a = '\n'.join(a)
df['MUST_CINSIYET'] = df['MUST_CINSIYET'].fillna(a)
####################################"""


# df.to_csv('modified_kk.csv', index=False, sep=';',encoding='windows-1254')##Yeni csvye yazdım.

##########KATEGORIK-NUMERIK-DATE AYRIMI####################
def get_categorical_and_numeric_columns(df, list_of_variables=None):
    x = df.dtypes
    x = pd.DataFrame(x, columns=['types'])
    if (list_of_variables != None):
        x = x[x.index.isin(list_of_variables)]

    binary_kategorik = []
    int_col = x[(x['types'] == 'int64')].index
    int_col = int_col.dropna()
    int_col = int_col.tolist()

    for i in int_col:
        n = len(pd.unique(df[i]))
        # nonUnique > 10 ise numerik olarak al. isimden de filtrele.
        if n < KATEGORIK_SARTI and i.find("SAYI") == -1 and i.find("SY") == -1 and i.find("TUT") == -1:
            binary_kategorik.append(i)

    for i in binary_kategorik:
        int_col.remove(i)

    num_cols = x[(x['types'] == 'float64')].index
    num_cols = num_cols.dropna()
    num_cols = num_cols.tolist()
    for i in int_col:
        num_cols.append(i)

    cat_cols = x[(x['types'] == 'object')].index
    cat_cols = cat_cols.dropna()
    cat_cols = cat_cols.tolist()
    for i in binary_kategorik:
        cat_cols.append(i)

    date_cols = x[(x['types'] == 'datetime64[ns]')].index
    date_cols = date_cols.dropna()
    date_cols = date_cols.tolist()

    print('Kategorik ve nümerik veriler döndürüldü.')

    return cat_cols, num_cols, date_cols


##  Csvde float değerleri . ile ayırdım ardından okuma yaptım.

cat, num, date = get_categorical_and_numeric_columns(df)


##########OUTLIER TEMIZLEME ISLEMLERI###########
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


###############BASIKLAMA UYGULA#############
def replace_with_thresholds(dataframe, columns_list):
    for variable in columns_list:
        low_limit, up_limit = outlier_thresholds(dataframe, variable)
        if low_limit < 0:
            low_limit = 0
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


outlier_variable_names = has_outliers(df, num)
# Silmek ya da baskılamak-OUTLIER ICIN.
replace_with_thresholds(df, outlier_variable_names)


# df.to_csv('modified_kk.csv', index=False, sep=';',encoding='windows-1254')##Yeni csvye yazdım.

###############VARYANS ANALIZI###################
def variance_analysis(dataframe, numerical_columns, threshold):
    variance_list = []
    for column_name in numerical_columns:
        if dataframe[column_name].var() <= threshold:
            variance_list.append(column_name)
    return variance_list


variance_remove_list = variance_analysis(df, num, 0.05)
df.drop(variance_remove_list, axis=1, inplace=True)
print('Varyans Analizi tamamlandı.')

# df.to_csv('modified_kk.csv', index=False, sep=';',encoding='windows-1254')##Yeni csvye yazdım.

tempdf = df.copy()
tempdf.drop('KK_ACIK_FLAG', axis=1, inplace=True)


#################YUKSEK KORALE KOLONLARIN AYRISMASI########################
def identify_correlated(df, threshold):
    matrix = df.corr().abs()
    mask = np.triu(np.ones_like(matrix, dtype=bool))
    reduced_matrix = matrix.mask(mask)
    to_drop = [c for c in reduced_matrix.columns if any(reduced_matrix[c] > threshold)]
    return to_drop


corr_drop_list = identify_correlated(tempdf, 0.80)
df.drop(corr_drop_list, axis=1, inplace=True)

# df.to_csv('modified_kk.csv', index=False, sep=';',encoding='windows-1254')##Yeni csvye yazdım.
############################################################################

##korelasyon grafigi
correlation = df.corr()
ax = sns.heatmap(correlation, center=0, cmap='RdBu_r')
l, r = ax.get_ylim()
ax.set_ylim(l + 0.5, r - 0.5)
plt.yticks(rotation=0)
plt.title('Correlation Matrix')
plt.show()

################# Target degısken Balans Analizi ###################
balans_oran = df['KK_ACIK_FLAG'].value_counts(normalize=True) * 100

sifirlar = df.loc[(df['KK_ACIK_FLAG'] == 0)]
birler = df.loc[(df['KK_ACIK_FLAG'] == 1)]

## 0.60-0.40 bir dagilim olusturulmak istendi. UNDERSAMPLING
major = (birler.shape[0] / 6) * 10
major = int(major)

sifirlar = sifirlar.sample(major)

temp = pd.concat([sifirlar, birler])
balans_oran_after = temp['KK_ACIK_FLAG'].value_counts(normalize=True) * 100

###Balanslama manuel yapıldı.
"""df.set_index('REF_TCKNVKN_ID', inplace=True)
tempdf.set_index('REF_TCKNVKN_ID', inplace=True)
X = tempdf.copy()
y = df['KK_ACIK_FLAG'].copy()

undersample = RandomUnderSampler(sampling_strategy=0.60)
# fit and apply the transform
X_over, y_over = undersample.fit_resample(X, y)
X_over.index = X.index[undersample.sample_indices_]


balans_oran_after = y_over.value_counts(normalize=True) * 100"""

########ENCODING##############
temp.set_index('REF_TCKNVKN_ID', inplace=True)
target = temp['KK_ACIK_FLAG'].copy()

temp.drop('KK_ACIK_FLAG', axis=1, inplace=True)

x = temp.dtypes
x = pd.DataFrame(x, columns=['types'])
cat_cols = x[(x['types'] == 'object')].index
cat_cols = cat_cols.dropna()
cat_cols = cat_cols.tolist()


### unique değer sayısı 2 ise label 2den fazla ise one hot kullan.

def one_hot_encoder(dataframe, categorical_columns, nan_as_category=False):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_columns, dummy_na=nan_as_category, drop_first=True)
    new_columns = [col for col in dataframe.columns if col not in original_columns]
    return dataframe, new_columns


temp, new_columns = one_hot_encoder(temp, cat_cols)
temp = pd.concat([temp, target], axis=1)

df = temp.copy()
df.reset_index(inplace=True)
# df.to_csv('modified_kk.csv', index=False, sep=';',encoding='windows-1254')

#df.to_csv('SON_modified_kk.csv', index=False, sep=';',encoding='windows-1254')
