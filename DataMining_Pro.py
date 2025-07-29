import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# تحميل البيانات
products = pd.read_csv("Extended_Products_Dataset__25_Products_ .csv")
invoices = pd.read_csv("Invoices_Dataset_for_Association_Rules.csv")

# -------------------------------
#1. التحليل الاستكشافي EDA
# -------------------------------

print("أول 5 صفوف من البيانات:")
print(products.head())

print("\nالقيم المفقودة:")
print(products.isnull().sum())

print("\nوصف إحصائي:")
print(products.describe())

# توزيع السعر والتقييم
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(products['Price'], kde=True)
plt.title("Distribution of Price")
plt.subplot(1, 2, 2)
sns.histplot(products['Rating'], kde=True)
plt.title("Distribution of Rating")
plt.tight_layout()
plt.savefig("eda_price_rating.png")

# -------------------------------
# معالجة البيانات
# -------------------------------

products.dropna(inplace=True)
categorical_cols = ['UsageType', 'MaterialType', 'ConnectivityType', 'SupplierCountry', 'Brand', 'Category', 'PriceCategory']
for col in categorical_cols:
    products[col] = LabelEncoder().fit_transform(products[col])