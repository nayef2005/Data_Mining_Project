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


# -------------------------------
# عنقدة KMeans
# -------------------------------

features_base = ['UsageType', 'MaterialType', 'ConnectivityType', 'PowerWatt', 'VolumeCm3', 'WeightKg', 'SupplierCountry', 'WarrantyYears', 'Stock', 'Rating', 'Brand', 'Category']
features_with_price = features_base + ['Price']

def apply_kmeans(data, n_clusters=5):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    preds = kmeans.fit_predict(data_scaled)
    score = silhouette_score(data_scaled, preds)
    print(f"Silhouette Score: {score:.3f}")
    return preds

print("\nعنقدة بدون السعر:")
products['Cluster_NoPrice'] = apply_kmeans(products[features_base])

print("\nعنقدة مع السعر:")
products['Cluster_WithPrice'] = apply_kmeans(products[features_with_price])


# -------------------------------
# تحليل قواعد الارتباط
# -------------------------------

# تحويل بيانات الفواتير إلى صيغة معاملات
invoice_product = invoices.groupby("InvoiceID")["ProductID"].apply(list)
te = TransactionEncoder()
te_ary = te.fit_transform(invoice_product)
df_tf = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_tf, min_support=0.03, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules[rules['confidence'] > 0.5].sort_values(by='lift', ascending=False).reset_index(drop=True)

# عرض أفضل 10 قواعد
print("\nTop 10 Association Rules by Lift:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# رسم علاقة lift مقابل 
plt.figure(figsize=(8, 6))
sns.scatterplot(data=rules, x='confidence', y='lift', size='support', hue='support', palette='viridis', legend=False)
plt.title("Association Rules: Confidence vs Lift")
plt.savefig("association_rules_plot.png")

# حفظ النتائج
products.to_csv("products_with_clusters.csv", index=False)

# حفظ قواعد الارتباط 
rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
rules.to_csv("association_rules.csv", index=False)