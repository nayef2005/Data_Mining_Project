import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- إعدادات أساسية ---
EVALUATION_FILE = "evaluations.csv"

# --- دوال مساعدة ---

# تحميل البيانات مع التخزين المؤقت لتحسين الأداء
@st.cache_data
def load_data():
    products = pd.read_csv("products_with_clusters.csv")
    rules = pd.read_csv("association_rules.csv")
    # تحويل النصوص إلى قوائم
    rules['antecedents'] = rules['antecedents'].apply(json.loads)
    rules['consequents'] = rules['consequents'].apply(json.loads)
    return products, rules

# دالة لحفظ التقييم
def save_evaluation(product_id, rec_type, rating):
    if not os.path.exists(EVALUATION_FILE):
        eval_df = pd.DataFrame(columns=["ProductID", "RecommendationType", "Rating"])
    else:
        eval_df = pd.read_csv(EVALUATION_FILE)
    
    new_entry = pd.DataFrame([{"ProductID": product_id, "RecommendationType": rec_type, "Rating": rating}])
    eval_df = pd.concat([eval_df, new_entry], ignore_index=True)
    eval_df.to_csv(EVALUATION_FILE, index=False)

# --- تحميل البيانات ---
products, rules = load_data()

# --- واجهة المستخدم ---
st.set_page_config(layout="wide")
st.title("نظام توصية مع دراسة تجريبية لتأثير السعر")
st.markdown("يهدف هذا النظام إلى تحليل جودة التوصيات الناتجة عن *قواعد الارتباط* و*العنقدة، مع دراسة تأثير **السعر* كعامل مؤثر في عملية العنقدة.")

# --- اختيار المنتج ---
product_name = st.selectbox("اختر منتجًا لبدء التحليل:", sorted(products['ProductName'].unique()))
product_row = products[products['ProductName'] == product_name].iloc[0]
prod_id = product_row['ProductID']
cluster_np = product_row['Cluster_NoPrice']
cluster_wp = product_row['Cluster_WithPrice']

# --- عرض معلومات المنتج ---
st.markdown("معلومات المنتج المحدد")
st.table(product_row[['Brand', 'Category', 'Price', 'Rating', 'Stock']].to_frame().T)

# --- تقسيم الواجهة إلى أعمدة ---
col1, col2, col3 = st.columns(3)

# --- العمود الأول: توصيات قواعد الارتباط ---
with col1:
    st.subheader("🔗 توصيات قواعد الارتباط")
    related_rules = rules[rules['antecedents'].apply(lambda x: prod_id in x)]
    recs = set(con for cons in related_rules['consequents'] for con in cons)
    
    if recs:
        rec_df = products[products['ProductID'].isin(recs)][['ProductID', 'ProductName', 'Brand', 'Price']]
        st.dataframe(rec_df.reset_index(drop=True))
        
        # قسم التقييم
        with st.expander("تقييم جودة توصيات قواعد الارتباط"):
            rating_ar = st.slider("ما مدى جودة هذه التوصيات؟ (0=سيئة, 100=ممتازة)", 0, 100, 50, key="ar_rating")
            if st.button("حفظ تقييم قواعد الارتباط", key="ar_btn"):
                save_evaluation(prod_id, "AssociationRules", rating_ar)
                st.success("تم حفظ تقييمك بنجاح!")
    else:
        st.info("لا توجد توصيات حالية بناءً على قواعد الارتباط لهذا المنتج.")

# --- العمود الثاني: توصيات العنقود (بدون السعر) ---
with col2:
    st.subheader("توصيات العنقود (بدون السعر)")
    same_cluster_np = products[(products['Cluster_NoPrice'] == cluster_np) & (products['ProductID'] != prod_id)]
    
    if not same_cluster_np.empty:
        st.dataframe(same_cluster_np[['ProductName', 'Brand', 'Rating']].reset_index(drop=True))
        
        # قسم التقييم
        with st.expander("تقييم جودة توصيات العنقود (بدون السعر)"):
            rating_np = st.slider("ما مدى جودة هذه التوصيات؟", 0, 100, 50, key="np_rating")
            if st.button("حفظ تقييم العنقود (بدون السعر)", key="np_btn"):
                save_evaluation(prod_id, "Cluster_NoPrice", rating_np)
                st.success("تم حفظ تقييمك بنجاح!")
    else:
        st.info("لا توجد منتجات مشابهة في هذا العنقود.")

# --- العمود الثالث: توصيات العنقود (مع السعر) ---
with col3:
    st.subheader("توصيات العنقود (مع السعر)")
    same_cluster_wp = products[(products['Cluster_WithPrice'] == cluster_wp) & (products['ProductID'] != prod_id)]
    
    if not same_cluster_wp.empty:
        st.dataframe(same_cluster_wp[['ProductName', 'Brand', 'Price', 'Rating']].reset_index(drop=True))
        
        # قسم التقييم
        with st.expander("تقييم جودة توصيات العنقود (مع السعر)"):
            rating_wp = st.slider("ما مدى جودة هذه التوصيات؟", 0, 100, 50, key="wp_rating")
            if st.button("حفظ تقييم العنقود (مع السعر)", key="wp_btn"):
                save_evaluation(prod_id, "Cluster_WithPrice", rating_wp)
                st.success("تم حفظ تقييمك بنجاح!")
    else:
        st.info("لا توجد منتجات مشابهة في هذا العنقود.")

st.divider()

# --- الجزء الثاني: تحليل الدراسة التجريبية ---
st.header("تحليل نتائج الدراسة التجريبية")

if os.path.exists(EVALUATION_FILE):
    eval_df = pd.read_csv(EVALUATION_FILE)
    
    # حساب متوسط التقييمات
    avg_ratings = eval_df.groupby('RecommendationType')['Rating'].mean().reset_index()
    avg_ratings = avg_ratings.sort_values(by="Rating", ascending=False)

    st.markdown("متوسط تقييم الجودة لكل نوع من التوصيات:")
    
    fig, ax = plt.subplots()
    sns.barplot(data=avg_ratings, x='RecommendationType', y='Rating', ax=ax, palette="viridis")
    ax.set_xlabel("نوع التوصية")
    ax.set_ylabel("متوسط التقييم (من 100)")
    ax.set_title("مقارنة متوسط جودة التوصيات")
    st.pyplot(fig)
    
    st.markdown("بناءً على التقييمات التي تم جمعها، يمكن استنتاج ما يلي:")
    if not avg_ratings.empty:
        best_method = avg_ratings.iloc[0]
        st.success(f"*الاستنتاج الأولي:* الطريقة الأعلى تقييمًا حتى الآن هي *{best_method['RecommendationType']}* بمتوسط تقييم يبلغ *{best_method['Rating']:.2f}*.")
    
    with st.expander("عرض جميع التقييمات المسجلة"):
        st.dataframe(eval_df)
else:
    st.info("لم يتم تسجيل أي تقييمات بعد. استخدم الأزرار أعلاه لتقييم جودة التوصيات.")