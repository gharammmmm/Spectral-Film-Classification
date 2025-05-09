import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from imblearn.over_sampling import SMOTE  # قراءة الملفات النصية مع معالجة الأخطاء

try:
    kodak = pd.read_csv('kodak.txt', sep='\t', header=None)
    fuji = pd.read_csv('fuji.txt', sep='\t', header=None)
    agfa = pd.read_csv('agfa.txt', sep='\t', header=None)
    unknown = pd.read_csv('inconnu.txt', sep='\t', header=None)
    print("تم تحميل الملفات بنجاح.")
except Exception as e:
    print(f"حدث خطأ أثناء تحميل الملفات: {e}")  # عدد الأعمدة في كل ملف قبل الدمج
print("عدد الأعمدة في ملف Fuji:", fuji.shape[1])  # هذا سيطبع عدد الأعمدة في ملف Fuji
print("عدد الأعمدة في ملف Agfa:", agfa.shape[1])  # هذا سيطبع عدد الأعمدة في ملف Agfa
print("عدد الأعمدة في ملف Kodak:", kodak.shape[1])  # هذا سيطبع عدد الأعمدة في ملف Kodak
print("عدد الأعمدة في ملف unknown:", unknown.shape[1])

# عدد السطور في كل ملف قبل الدمج
print("عدد السطور في ملف Fuji:", fuji.shape[0])  # هذا سيطبع عدد السطور في ملف Fuji
print("عدد السطور في ملف Agfa:", agfa.shape[0])  # هذا سيطبع عدد السطور في ملف Agfa
print("عدد السطور في ملف Kodak:", kodak.shape[0])  # هذا سيطبع عدد السطور في ملف Kodak
print("عدد السطور في ملف unknown:", unknown.shape[0])  # هذا سيطبع عدد السطور في ملف unknown

# إضافة عمود الفئة لكل مجموعة بيانات
fuji['Label'] = 'Fuji'
agfa['Label'] = 'Agfa'
kodak['Label'] = 'Kodak'

# دمج جميع البيانات في ملف واحد
data = pd.concat([fuji, agfa, kodak], axis=0, ignore_index=True)
# عرض معلومات عن البيانات
print("\nمعلومات البيانات:")
print(data.info())

# التحقق من القيم المفقودة
print("\nعدد القيم المفقودة:")
print(data.isnull().sum())

# تطبيق SMOTE لموازنة الفئات
print("\nعدد الملاحظات لكل فئة قبل SMOTE:")
print(data['Label'].value_counts())
X = data.iloc[:, :-1].values
y = data['Label'].values

smote = SMOTE(
    sampling_strategy={'Fuji': 100, 'Agfa': 119, 'Kodak': 128},
    random_state=42
)
X_resampled, y_resampled = smote.fit_resample(X, y)
# إنشاء DataFrame من البيانات المعاد توازنها
balanced_data = pd.DataFrame(X_resampled)
balanced_data['Label'] = y_resampled

print("\nعدد الملاحظات لكل فئة بعد SMOTE:")
print(balanced_data['Label'].value_counts())
# استخراج الميزات والتسميات من البيانات المعاد توازنها
X = balanced_data.iloc[:, :-1].values
y = balanced_data['Label'].values
# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\nحجم مجموعة التدريب:", len(X_train))
print("حجم مجموعة الاختبار:", len(X_test))  # معالجة القيم المفقودة
imputer = SimpleImputer(strategy="mean")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)  # تقييس البيانات
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# حفظ الـ scaler
joblib.dump(scaler, 'scaler.pkl')


def plot_accuracy_comparison(accuracies, title):
    plt.figure(figsize=(8, 5))
    names = list(accuracies.keys())
    values = list(accuracies.values())
    sns.barplot(x=names, y=values, palette="viridis")
    plt.title(title)
    plt.ylabel("الدقة (%)")
    plt.xlabel("الخوارزمية")
    plt.ylim(0, 100)  # لضمان توحيد المقياس
    plt.show()


# خوارزميات التصنيف
classifiers = {
    "Decision Tree (CART)": DecisionTreeClassifier(random_state=42),
    "LDA": LDA(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "SVM": SVC(kernel='linear', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# متغيرات لتخزين أفضل نموذج
best_accuracy = 0
best_model = None
best_model_name = ""

# حفظ الدقة لرسومات المقارنة
accuracies_original = {}
accuracies_pca = {}
accuracies_lda = {}

# التصنيف على البيانات الأصلية
print("\nنتائج التصنيف على البيانات الأصلية:")
for name, clf in classifiers.items():
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracies_original[name] = accuracy
    print(f"{name} - الدقة: {accuracy:.2f}%")
    print("مصفوفة الارتباك:\n", confusion_matrix(y_test, y_pred), "\n")

    # تحديث أفضل نموذج
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = clf
        best_model_name = f"{name} (البيانات الأصلية)"

# رسم مقارنة الدقة للبيانات الأصلية
plot_accuracy_comparison(accuracies_original, "مقارنة دقة النماذج - البيانات الأصلية")  # التصنيف بعد PCA
pca = PCA(n_components=50)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# تدريب LDA بعد PCA
lda = LDA()
lda.fit(X_train_pca, y_train)

# حفظ LDA والنماذج الأخرى
joblib.dump(lda, 'best_lda.pkl')
joblib.dump(pca, 'best_pca.pkl')

print("\nنتائج التصنيف بعد PCA:")
for name, clf in classifiers.items():
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred) * 100
    accuracies_pca[name] = accuracy
    print(f"{name} - الدقة بعد PCA: {accuracy:.2f}%")
    print("مصفوفة الارتباك:\n", confusion_matrix(y_test, y_pred), "\n")

    # تحديث أفضل نموذج
    if accuracy > best_accuracy or (accuracy == 100.00 and name == 'LDA'):  # تخصيص LDA في حالة التساوي
        best_accuracy = accuracy
        best_model = clf
        best_model_name = f"{name} (بعد PCA)"

# رسم مقارنة الدقة بعد PCA
plot_accuracy_comparison(accuracies_pca, "مقارنة دقة النماذج - بعد PCA")

# حفظ LDA بعد PCA إذا كانت هي الأفضل
lda_after_pca = LDA()
lda_after_pca.fit(X_train_pca, y_train)
joblib.dump(lda_after_pca, 'best_lda_after_pca.pkl')
print("✅ تم حفظ LDA بعد PCA: best_lda_after_pca.pkl")

# حفظ أفضل نموذج مع إعطاء الأولوية لـ LDA
if best_model_name == "LDA (بعد PCA)":
    joblib.dump(best_model, "best_model.pkl")
    print(f"✅ تم حفظ LDA كأفضل نموذج: {best_model_name}")
elif best_model_name == "QDA (بعد PCA)":
    joblib.dump(best_model, "best_model.pkl")
    print(f"✅ تم حفظ QDA كأفضل نموذج: {best_model_name}")

joblib.dump(scaler, 'scaler.pkl')
print(f"\n✅ أفضل نموذج: {best_model_name} - دقة: {best_accuracy:.2f}%")
print("- تم حفظ best_model.pkl و scaler.pkl")

if "(بعد PCA)" in best_model_name:
    joblib.dump(pca, 'best_pca.pkl')
    print("- تم حفظ PCA: best_pca.pkl")

if "(بعد FDA)" in best_model_name:
    joblib.dump(top_6_indices, 'best_fda_indices.pkl')
    joblib.dump(fisher_scores, 'fda_scores.pkl')
    print("- تم حفظ ملفات FDA")


# 1. حساب Fisher Scores لكل طول موجي
def calculate_fisher_scores(X, y):
    classes = np.unique(y)
    n_features = X.shape[1]
    fisher_scores = np.zeros(n_features)

    for i in range(n_features):
        overall_mean = np.mean(X[:, i])
        numerator = 0
        denominator = 0

        for c in classes:
            X_c = X[y == c, i]
            mean_c = np.mean(X_c)
            var_c = np.var(X_c, ddof=1)

            n_c = len(X_c)
            numerator += n_c * (mean_c - overall_mean) ** 2
            denominator += (n_c - 1) * var_c

        fisher_scores[i] = numerator / denominator if denominator != 0 else 0

    return fisher_scores


# 2. حساب درجات Fisher واستخراج أفضل 6 أطوال
fisher_scores = calculate_fisher_scores(X_train_scaled, y_train)
top_6_indices = np.argsort(fisher_scores)[-6:][::-1]  # أفضل 6 أطوال
print("أفضل 6 أطوال موجية (Fisher Score):", top_6_indices)

# حفظ المؤشرات ودرجات Fisher
joblib.dump(top_6_indices, 'best_fda_indices.pkl')
joblib.dump(fisher_scores, 'fda_scores.pkl')

# 3. استخراج البيانات المميزة فقط
X_train_fisher = X_train_scaled[:, top_6_indices]
X_test_fisher = X_test_scaled[:, top_6_indices]

# 4. تطبيق جميع الخوارزميات على الأطوال المختارة
results = {}
for name, clf in classifiers.items():
    clf.fit(X_train_fisher, y_train)
    y_pred = clf.predict(X_test_fisher)
    accuracy = accuracy_score(y_test, y_pred) * 100
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'confusion_matrix': cm}

    print(f"\n{name} - الدقة بعد FDA: {accuracy:.2f}%")
    print("مصفوفة الارتباك:\n", cm)

# 5. حفظ أفضل نموذج (مثل LDA أو QDA)
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = classifiers[best_model_name]
joblib.dump(best_model, f'best_model_fisher_{best_model_name}.pkl')

# 6. رسم نتائج الدقة
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [results[name]['accuracy'] for name in results])
plt.title("مقارنة دقة الخوارزميات بعد تطبيق FDA")
plt.ylabel("الدقة (%)")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.show()  # بعد قسمة PCA والتدريب، قم بإضافة هذا الكود لحفظ أفضل نموذج

# تحديد أفضل نموذج (LDA بعد PCA)
best_model_name = "LDA (بعد PCA)"
best_model = LDA()  # إنشاء نموذج LDA جديد
best_model.fit(X_train_pca, y_train)  # تدريبه على بيانات PCA

# حفظ المكونات الضرورية
import joblib

# 1. حفظ أفضل نموذج
joblib.dump(best_model, 'best_model.pkl')

# 2. حفظ PCA
joblib.dump(pca, 'pca_model.pkl')

# 3. حفظ Scaler
joblib.dump(scaler, 'scaler.pkl')

print("تم حفظ المكونات بنجاح:")
print("- best_model.pkl (نموذج LDA المدرب)")
print("- pca_model.pkl (نموذج PCA المدرب)")
print("- scaler.pkl (نموذج التطبيع)")
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. تحميل النماذج المحفوظة
try:
    scaler = joblib.load("scaler.pkl")  # نموذج التقييس
    pca = joblib.load("best_pca.pkl")  # نموذج PCA
    lda_model = joblib.load("best_model.pkl")  # نموذج LDA المدرب بعد PCA
except Exception as e:
    raise ValueError(f"❌ خطأ في تحميل النماذج: {e}")

# 2. تطبيق خطوات المعالجة على بيانات الاختبار (بنفس ترتيب التدريب)
X_test_scaled = scaler.transform(X_test)  # التقييس
X_test_pca = pca.transform(X_test_scaled)  # تطبيق PCA

# 3. التنبؤ باستخدام LDA
y_pred = lda_model.predict(X_test_pca)

# 4. تقييم النموذج
accuracy = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)

# 5. عرض النتائج
print("\n" + "=" * 50)
print(f"✅ دقة النموذج (LDA بعد PCA): {accuracy:.2f}%")
print("\n📊 مصفوفة الارتباك:")
print(cm)
print("\n🔍 المقارنة:")
for true, pred in zip(y_test, y_pred):
    print(f"الحقيقي: {true} ← المتنبأ: {pred}")
print("=" * 50)
import joblib
import numpy as np


def predict_unknown_samples(unknown_data):
    """
    تقوم هذه الدالة بالتنبؤ بالبيانات غير المعروفة باستخدام LDA بعد PCA
    :param unknown_data: بيانات غير معروفة (DataFrame أو numpy array)
    :return: تنبؤات النموذج
    """
    # 1. تحميل النماذج المحفوظة
    try:
        print("\n🔍 جاري تحميل النماذج...")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("best_pca.pkl")
        lda_model = joblib.load("best_model.pkl")
        print("✅ تم تحميل النماذج بنجاح (Scaler, PCA, LDA)")
    except Exception as e:
        raise ValueError(f"❌ خطأ في تحميل النماذج: {e}")

    # 2. تحويل البيانات إلى numpy array إذا كانت DataFrame
    if hasattr(unknown_data, 'values'):
        X_unknown = unknown_data.values
    else:
        X_unknown = np.array(unknown_data)

    # 3. تطبيق خطوات المعالجة (بنفس ترتيب التدريب)
    try:
        print("\n⚙️ جاري معالجة البيانات...")
        X_unknown_scaled = scaler.transform(X_unknown)  # التقييس
        X_unknown_pca = pca.transform(X_unknown_scaled)  # تطبيق PCA
        print("✅ تم تطبيق التقييس و PCA بنجاح")
    except Exception as e:
        raise ValueError(f"❌ خطأ في معالجة البيانات: {e}")

    # 4. التنبؤ باستخدام LDA
    try:
        print("\n🔮 جاري التنبؤ...")
        y_pred = lda_model.predict(X_unknown_pca)
        print("✅ تم الانتهاء من التنبؤ بنجاح")
    except Exception as e:
        raise ValueError(f"❌ خطأ في التنبؤ: {e}")

    # 5. عرض النتائج
    print("\n🎯 نتائج التنبؤ:")
    print("=" * 40)
    for i, pred in enumerate(y_pred, 1):
        print(f"العينة {i}: → الفئة {pred}")
    print("=" * 40)

    return y_pred


# كيفية الاستخدام:
# predictions = predict_unknown_samples(unknown_data)
predictions = predict_unknown_samples(unknown) 