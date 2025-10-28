# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import os

# ---------------------------
# Page title
# ---------------------------
st.title("Bangla Newspaper Dataset Analysis")

# ---------------------------
# Download dataset if not exists
# ---------------------------
file_id = "1KYEuvvLLV7a0IRTaOU-U5X9Ysf6wN7W8"
output = "newspaper.json"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(output):
    st.write("Downloading dataset...")
    gdown.download(url, output, quiet=False)
else:
    st.write("Dataset already exists.")

# ---------------------------
# Load dataset
# ---------------------------
df1 = pd.read_json(output)

# ---------------------------
# Clean dataset
# ---------------------------
df1 = df1.drop_duplicates(subset='content', keep='first')
df1 = df1[~df1['category'].isin(['bangladesh', 'opinion'])]
df1 = df1.reset_index(drop=True)

# ---------------------------
# Show dataset info
# ---------------------------
st.subheader("Cleaned Dataset Info")
st.text(df1.info())
st.subheader("Sample Data")
st.dataframe(df1.head())

# ---------------------------
# Plot category counts
# ---------------------------
category_counts = df1['category'].value_counts()
fig, ax = plt.subplots(figsize=(12,6))
category_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title("Number of Articles per Category (Cleaned Dataset)", fontsize=16)
ax.set_xlabel("Category", fontsize=14)
ax.set_ylabel("Number of Articles", fontsize=14)
plt.xticks(rotation=45)

st.subheader("Category Distribution")
st.pyplot(fig)

# app_dataset2.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

# ---------------------------
# Page title
# ---------------------------
st.title("Bangla Newspaper Dataset 2 Analysis")

# ---------------------------
# Download dataset if not exists
# ---------------------------
file_id = "1OtPy0n-LsceeDPJI5yfK8ekVneHHkeLR"
output = "Bangla_Newspaper_Article_Dataset.csv"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(output):
    st.write("Downloading Dataset 2 (this may take a while)...")
    gdown.download(url, output, quiet=False)
else:
    st.write("Dataset 2 already exists locally. Skipping download.")

# ---------------------------
# Load dataset
# ---------------------------
df2 = pd.read_csv(output, encoding='utf-8')

st.subheader("Dataset Info")
st.text(df2.info())
st.subheader("Sample Data")
st.dataframe(df2.head())

# ---------------------------
# Plot category distribution
# ---------------------------
category_counts2 = df2['category'].value_counts()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=category_counts2.index, y=category_counts2.values, palette='magma', ax=ax)

# Add frequency labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge')

ax.set_title('Dataset 2 News Category Distribution')
ax.set_xlabel('Category')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()

st.subheader("Category Distribution")
st.pyplot(fig)
# app_data_processing.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Page title
# ---------------------------
st.title("Bangla News Data Processing & Visualization")

# ---------------------------
# Assume df1 and df2 are already loaded (from previous apps)
# ---------------------------
# df1 = ... (Dataset 1)
# df2 = ... (Dataset 2)

st.subheader("Initial Dataset Info")
st.write("Dataset 1 info:")
st.text(df1.info())
st.dataframe(df1.head())

st.write("Dataset 2 info:")
st.text(df2.info())
st.dataframe(df2.head())

# ---------------------------
# Data Cleaning
# ---------------------------
df1_clean = df1.drop(columns=['author', 'category_bn', 'modification_date', 'tag', 'comment_count'], errors='ignore')
df1_clean.rename(columns={'url': 'source'}, inplace=True)
df1_clean = df1_clean.dropna().reset_index(drop=True)
df1_clean = df1_clean.drop_duplicates(subset=['content'], keep='first').reset_index(drop=True)
df1_clean['category'] = df1_clean['category'].replace('life-style', 'lifestyle')
df1_clean = df1_clean[~df1_clean['category'].isin(['bangladesh', 'opinion'])]

df2_clean = df2.dropna().reset_index(drop=True)
df2_clean = df2_clean.drop_duplicates(subset=['content'], keep='first').reset_index(drop=True)

# ---------------------------
# Combine datasets
# ---------------------------
df = pd.concat([df1_clean, df2_clean], ignore_index=True)
st.subheader("Combined Dataset Info")
st.text(df.info())
st.write("Category counts before balancing:")
st.write(df['category'].value_counts())

# ---------------------------
# Plot histogram of categories
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 6))
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values, palette='magma', ax=ax)
ax.set_title('News Category Distribution (Before Balancing)')
ax.set_xlabel('Category')
ax.set_ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# Balance classes
# ---------------------------
target_size = 5000
balanced_dfs = []

for category, group in df.groupby('category'):
    if len(group) > target_size:
        sampled = group.iloc[-target_size:]  # keep last rows
    else:
        sampled = group
    balanced_dfs.append(sampled)

df_balanced = pd.concat(balanced_dfs, ignore_index=True)
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

st.subheader("Balanced Dataset Info")
st.text(df_balanced.info())
st.write("Category counts after balancing:")
st.write(df_balanced['category'].value_counts())
st.write("Shape:", df_balanced.shape)

# ---------------------------
# Plot histogram after balancing
# ---------------------------
fig2, ax2 = plt.subplots(figsize=(10, 6))
category_counts_bal = df_balanced['category'].value_counts()
sns.barplot(x=category_counts_bal.index, y=category_counts_bal.values, palette='magma', ax=ax2)
ax2.set_title('News Category Distribution (After Balancing)')
ax2.set_xlabel('Category')
ax2.set_ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig2)

# app_clean_text.py
import streamlit as st
import pandas as pd
import re
import unicodedata

st.title("Bangla Text Cleaning & Tokenization")

# ---------------------------
# Assume df_balanced is already created from previous processing step
# ---------------------------
st.subheader("Balanced Dataset Info Before Cleaning")
st.text(df_balanced.info())
st.dataframe(df_balanced.head(5))

# ---------------------------
# Bangla Stopword List
# ---------------------------
raw_stopwords = [
    "অবশ্য", "অন্তত", "অথবা", "অথচ", "অর্থাত", "অন্য", "আজ", "আছে", "আপনার", "আপনি", "আবার", "আমরা", "আমাকে", "আমাদের", "আমার", "আমি", "আরও", "আর",
    "আগে", "আগেই","আগামী", "অবধি", "অনুযায়ী", "আদ্যভাগে", "এই", "একই", "এককে", "একটি", "এখন", "এখনও", "এখানে", "এখানেই", "এটি", "এটা", "এটাই", "এতটাই", "এবং", "একবার",
    "এবার", "এদের", "এঁদের", "এমন", "এমনকী", "এল", "এর", "এরা", "এঁরা", "এস", "এত", "এতে", "এসে", "একে", "এ", "ঐ", "ই", "ইহা", "ইত্যাদি", "উনি", "উপর", "উপরে",
    "উচিত", "ও", "ওই", "ওর", "ওরা", "ওঁর", "ওঁরা", "ওকে", "ওদের", "ওঁদের", "ওখানে", "কত", "কবে", "করতে", "কয়েক", "কয়েকটি", "করবে", "করলেন", "করার", "কারও",
    "করা", "করি", "করিয়ে", "করাই", "করলে", "করিতে", "করিয়া", "করেছিলেন", "করছে", "করছেন", "করেছেন", "করেছে", "করেন", "করবেন", "করায়", "করে", "করেই", "কাছ", "কাছে",
    "কারণ", "কিছু", "কিছুই", "কিন্তু", "কিংবা", "কি", "কী", "কেউ", "কেউই", "কাউকে", "কেন", "কে", "কোনও", "কোনো", "কোন", "কখনও", "ক্ষেত্রে", "খুব", "গুলি", "গিয়ে",
    "গিয়েছে", "গেছে", "গেল", "গেলে", "গোটা", "চলে", "চেয়ে", "ছাড়া", "ছাড়াও", "ছিলেন", "ছিল", "জন্য", "জানা", "ঠিক", "তিনি", "তিনঐ", "তিনিও", "তখন", "তবে", "তবু",
    "তাঁদের", "তাঁহারা", "তাঁরা", "তাঁর", "তাঁকে", "তাই", "তেমন", "তাকে", "তাহা", "তাহাতে", "তাহার", "তাদের", "তারপর", "তারা", "তারৈ", "তার", "তাহলে", "তা", "তাও", "তাতে",
    "তো", "তত", "তুমি", "তোমার", "তথা", "থাকে", "থাকা", "থাকায়", "থেকে", "থেকেও", "থাকবে", "থাকেন", "থাকবেন", "থেকেই", "দিকে", "দিতে", "দিয়ে", "দিয়েছে", "দিয়েছেন",
    "দু", "দুটি", "দুটো", "দেয়", "দেয়া", "দেওয়া", "দেওয়ার", "দেখা", "দেখে", "দেখতে", "দ্বারা", "ধরে", "ধরা", "নয়", "নানা", "না", "নাকি", "নাগাদ", "নিতে", "নিজে","কাজে",
    "নিজেই", "নিজের", "নিজেদের", "নিয়ে", "নেওয়া", "নেওয়ার", "নেই", "নাই", "পক্ষে", "পর্যন্ত", "পাওয়া", "পারেন", "পারি", "পারে", "পরে", "পরেই", "পরেও", "পর", "পেয়ে", "প্রতি",
    "প্রভৃতি", "প্রায়", "ফের", "ফলে", "ফিরে", "ব্যবহার", "বলতে", "বললেন", "বলেছেন", "বলল", "বলা", "বলেন", "বলে", "বহু", "বসে", "বার", "বা", "বিনা", "বরং", "বদলে",
    "বাদে", "বিশেষ", "বিভিন্ন", "বিষয়টি", "ব্যবহার", "ব্যাপারে", "ভাবে", "ভাবেই", "মধ্যে", "মধ্যেই", "তোমাদের", "তোমরা", "মানুষ", "মানুষের", "মধ্যেও", "মধ্যভাগে", "মাধ্যমে", "মাঝে",
    "মতোই", "মোটেই", "যখন", "যদি", "যদিও", "যাবে", "যায়", "যাকে", "যাওয়া", "যাওয়ার", "যত", "যতটা", "যা", "যার", "যারা", "যাঁর", "যাঁরা", "যাদের", "যান", "যাচ্ছে",
    "যেতে", "যাতে", "যেন", "যেমন", "যেখানে", "যিনি", "যে", "রেখে", "রাখা", "রয়েছে", "রকম", "শুধু", "সঙ্গে", "সঙ্গেও", "সমস্ত", "সব", "সবার", "সহ", "সুতরাং", "সহিত",
    "সেই", "সেটা", "সেটি", "সেটাই", "সেটাও", "সম্প্রতি", "সেখান", "সেখানে", "সে", "স্পষ্ট", "স্বয়ং", "হইতে", "হইবে", "হৈলে", "হইয়া", "হচ্ছে", "হত", "কোনটি", "হতে", "হতেই",
    "হবে", "হবেন", "হয়েছিল", "হয়েছে", "হয়েছেন", "হয়ে", "হয়নি", "হয়", "হয়েই", "হয়তো", "হল", "হলে", "হলেই", "হলেও", "হলো", "হিসাবে", "হওয়া", "হওয়ার", "হওয়ায়", "হন",
    "হোক", "দেখা যায়", "শোনা যায়", "গত", "নিয়ে", "যায়", "হয়ে", "কথা", "দেওয়া", "কাজ", "তৈরি", "জানান", "দিয়ে", "জানিয়েছে", "০", "১", "১০", "১১", "১২", "১৩",
    "১৪", "২", "৩", "৪", "৫", "৬", "৭", "৮", "৯", "১১", "১২", "১৩", "১৪", "১৫", "১৬", "১৭", "১৮", "১৯", "২০","২১", "২২", "২৩", "২৪", "২৫", "২৬", "২৭", "২৮", "২৯", "৩০",
    "৩১", "৩২", "৩৩", "৩৪", "৩৫", "৩৬", "৩৭", "৩৮", "৩৯", "৪০","৪১", "৪২", "৪৩", "৪৪", "৪৫", "৪৬", "৪৭", "৪৮", "৪৯", "৫০","৫১", "৫২", "৫৩", "৫৪", "৫৫", "৫৬", "৫৭", "৫৮", "৫৯", "৬০",
    "৬১", "৬২", "৬৩", "৬৪", "৬৫", "৬৬", "৬৭", "৬৮", "৬৯", "৭০","৭১", "৭২", "৭৩", "৭৪", "৭৫", "৭৬", "৭৭", "৭৮", "৭৯", "৮০",
    "৮১", "৮২", "৮৩", "৮৪", "৮৫", "৮৬", "৮৭", "৮৮", "৮৯", "৯০","৯১", "৯২", "৯৩", "৯৪", "৯৫", "৯৬", "৯৭", "৯৮", "৯৯", "১০০", "আলো", "এক", "একজন", "একটু", "ওপর", "খান", "কাজের থাকলে", "কারণে", "করো", "করুন", "কম","দিলেন","সাহায্য", "সুযোগ",
    "কমেছে", "চৌধুরী", "ছয়", "ছোট", "জানায়", "জানান", "জন", "চার", "যুক্ত", "ড:", "দিন", "দশ", "দুই", "নতুন", "শেষ", "নিলে", "নিন", "নয়", "পাবেন","মাত্র", "মতো",
    "পেতে", "পারবেন", "দূরে", "যেকোনো", "থাকলে", "সম্ভাবনা", "একটা", "শুভ", "গুরুত্বপূর্ণ", "থাকতে", "রাখুন", "খেতে", "ব্যক্তি", "ঘটনা", "প্রথম", "প্রধান", "প্রকাশ", "বছর", "বড়",
    "বেশ","বেশি", "মনে", "মো", "মোঃ", "প্রথম", "দ্বিতীয়", "তৃতীয়", "চতুর্থ", "পঞ্চম", "ষষ্ঠ", "সপ্তম", "অষ্টম", "নবম", "দশম", "একাদশ", "দ্বাদশ", "ত্রয়োদশ", "চতুর্দশ", "পঞ্চদশ", "ষোড়শ",
    "সপ্তদশ", "অষ্টাদশ", "ঊনবিংশ", "বিশতম", "১ম", "২য়", "৩য়", "৪র্থ", "৫ম", "৬ষ্ঠ", "৭ম", "৮ম", "৯ম", "১০ম", "১১তম", "১২তম", "১৩তম", "১৪তম", "১৫তম", "১৬তম",
    "১৭তম", "১৮তম", "১৯তম", "২০তম", "২১তম", "২২তম", "২৩তম", "২৪তম", "২৫তম", "২৬তম", "২৭তম", "২৮তম", "২৯তম", "৩০তম", "৩১তম", "৩২তম", "৩৩তম", "৩৪তম",
    "৩৫তম", "৩৬তম","৩৭তম", "৩৮তম", "৩৯তম", "৪০তম", "৪১তম", "৪২তম", "৪৩তম", "৪৪তম", "৪৫তম", "৪৬তম", "৪৭তম", "৪৮তম", "৪৯তম", "৫০তম", "শতাংশ", "চালু", "কোটি", "দেশের",
    "দেশ", "শত","হাজার", "লাখ", "কোটি", "মিলিয়ন", "বিলিয়ন", "বছর", "বছরের", "সুবিধা", "পাশাপাশি", "সেবা", "শত", "চেয়ারম্যান", "পরিচালক", "বিরুদ্ধে", "খবর", "অভিযোগ", "সালের",
    "হোসেন", "ধরনের","রহমান", "সালে", "অনুষ্ঠান", "অনুষ্ঠানে", "অনেক", "কমে", "দেন", "উপস্থিত", "সরকারের", "বেড়েছে", "দেশে", "শুরু", "হিসেবে", "মোট", "সাধারণ", "বিষয়ে", "সভায়", "অংশ",
    "এম", "আরো", "সুযোগ", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ", "বর্তমানে", "জাতীয়", "অ্যান্ড", "সম্পর্কে", "পড়ে", "সময়", "ক", "খ", "গ",
    "ঘ", "ঙ", "চ","ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "দ", "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ", "ক্ষ", "ড়", "ঢ়", "য়", "ৎ", "অ", "আ", "ই",
    "ঈ", "উ", "ঊ", "এ", "ঐ", "ও", "ঔ","সৃষ্টি", "হাতে", "এমনকি", "সামনে", "এসব", "তুলে", "গড়ে", "যেসব", "সেসব", "বন্ধ", "খোলা", "শুরু", "শেষ", "চেষ্টা", "সাফল্য", "সফলতা",
    "আশা","সামনে", "পিছনে", "পারবে", "ব্যবহারে", "হাতে", "হতে", "এগিয়ে", "এরপর", "তারপর", "অতঃপর","নেন", "শেষে", "শুরতে", "তুলে", "পারেননি", "কাল", "সবচেয়ে", "জানিয়েছেন",
    "জানিয়ে","জানিয়েছিল", "জানিয়েছিলে", "লিখেছেন", "লিখতে", "চাই", "শুরু", "শনিবার", "রবিবার", "সোমবার","মঙ্গলবার", "বুধবার", "বৃহঃস্পতিবার", "বৃহস্পতিবার", "শুক্রবার",  "সিন্ধান্ত", "আছেন",  "রাতে", "দুপুরে", "জানতে",
    "দাবি", "সাথে", "অবস্থায়", "নম্বর", "গ্রাম", "গ্রামের", "শহর", "শহরের","ব্যবস্থা", "বাড়ি", "বাড়িতে", "চালিয়ে", "জনকে", "ঘটনার", "সদর", "নম্বর", "সংবাদ", "পত্রিকা", "নিশ্চিত",
    "সহকারী", "ছেলে", "মেয়ে", "এসময়", "পাঠানো", "নিচের", "প্রতিটি", "সদস্য", "বাকি", "বাংলাদেশ", "ঘোষণা","ভূমিকা", "প্রয়োজন", "পরিমাণ", "অর্থ", "দাও", "নামে", "ঢাকা", "চট্টগ্রাম", "কুমিল্লা", "জানুয়ারি",
    "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগষ্ট", "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর","ডিসেম্বর", "বৈশাখ", "জৈষ্ঠ্য", "আষাঢ়", "শ্রাবণ", "ভাদ্র", "আশ্বিন", "কার্ত্তিক", "অগ্রহায়ন", "পৌষ",
    "মাঘ", "ফাল্গুন", "চৈত্র",  "নাম", "অন্যান্য", "রাখতে", "দেবে", "দাম", "নির্বাহী", "সহজ", "বলছে","সময়ে", "এসেছে", "উন্নত", "আপনাকে", "লাগান", "লাগিয়ে", "নিয়মিত",
    "জরুরি", "হক", "সাড়ে", "দায়িত্ব", "গ্রহণ", "ঘটনায়"
]
bangla_stopwords = set(unicodedata.normalize("NFC", word.strip()) for word in raw_stopwords)

# ---------------------------
# Cleaning Function
# ---------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""

    # Normalize
    text = unicodedata.normalize("NFC", text)

    # Remove non-Bangla characters
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    filtered = [t for t in tokens if t not in bangla_stopwords]

    return ' '.join(filtered)

# ---------------------------
# Apply cleaning on balanced dataset
# ---------------------------
st.write("Cleaning text... this may take a few seconds for large datasets.")
df_balanced['cleaned_content'] = df_balanced['content'].apply(clean_text)

st.subheader("Dataset After Cleaning")
st.dataframe(df_balanced[['content', 'cleaned_content']].head(10))
st.write("Total rows:", df_balanced.shape[0])

# app_class_stopwords.py
import streamlit as st
import pandas as pd
import re

st.title("Bangla Class-Specific Stopword Removal & Tokenization")

# ---------------------------
# Assume df_balanced is already cleaned (from previous step)
# ---------------------------
st.subheader("Dataset Before Class-Specific Stopword Removal")
st.dataframe(df_balanced[['category', 'cleaned_content']].head(5))

# ---------------------------
# Words to remove per class
# ---------------------------
class_word_map = {
    'technology': ['প্রতিমন্ত্রী', 'ব্যাংক', 'বাজারে','তথ্য', 'বাংলাদেশের', 'টাকা', 'নামের', 'সংখ্যা', 'পণ্য'],
    'economy': ['ছাত্রলীগের','প্রতিষ্ঠান', 'তথ্য', 'সরকার'],
    'entertainment': ['প্রতিমন্ত্রী', 'ব্যাংক', 'বাজারে','সামাজিক','পোস্ট'],
    'health': ['ঘন্টায়','গেছেন','দাঁড়িয়েছে','জানানো','বিজ্ঞপ্তি', 'সংখ্যা','বাইরে'],
    'education': ['সভাপতি','ইসলাম','শেখ', 'কমিটি', 'ছাত্রলীগের','অনুষ্ঠিত','তথ্য', 'এদিকে', 'সূত্র'],
    'crime': ['রাজধানী','ইসলাম','আলোকে', 'এলাকার'],
    'lifestyle': ['টাকা'],
    'environment': ['তথ্য'],
}

# ---------------------------
# Function to remove class-specific words
# ---------------------------
def remove_class_words(row):
    text = row['cleaned_content']
    class_name = row['category']

    if class_name in class_word_map:
        for word in class_word_map[class_name]:
            text = text.replace(word, '')
        text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------
# Apply to DataFrame
# ---------------------------
st.write("Removing class-specific words...")
df_balanced['cleaned_content'] = df_balanced.apply(remove_class_words, axis=1)

st.subheader("Dataset After Class-Specific Stopword Removal")
st.dataframe(df_balanced[['category', 'cleaned_content']].head(5))

# ---------------------------
# Tokenization
# ---------------------------
st.write("Tokenizing cleaned content into words...")
df_balanced['token_list'] = df_balanced['cleaned_content'].apply(lambda x: x.split())

st.subheader("Dataset with Token Lists")
st.dataframe(df_balanced[['category', 'token_list']].head(5))


# app_unique_words.py
import streamlit as st
from collections import defaultdict

st.title("Unique Words per Bangla News Category")

# ---------------------------
# Assume df_balanced has a 'token_list' column from previous step
# ---------------------------
st.subheader("Tokenized Dataset Sample")
st.dataframe(df_balanced[['category', 'token_list']].head(5))

# ---------------------------
# Step 1: Collect sets of unique words per category
# ---------------------------
category_word_sets = defaultdict(set)

for category in df_balanced['category'].unique():
    tokens = df_balanced[df_balanced['category'] == category]['token_list']
    all_tokens = [token for token_list in tokens for token in token_list]
    category_word_sets[category] = set(all_tokens)

# ---------------------------
# Step 2: Create a global word-to-category mapping
# ---------------------------
word_category_count = defaultdict(set)

for category, word_set in category_word_sets.items():
    for word in word_set:
        word_category_count[word].add(category)

# ---------------------------
# Step 3: Keep only words that appear in a single category
# ---------------------------
unique_words_by_category = defaultdict(list)

for word, categories in word_category_count.items():
    if len(categories) == 1:
        category = list(categories)[0]
        unique_words_by_category[category].append(word)

# ---------------------------
# Step 4: Show results in Streamlit
# ---------------------------
st.subheader("Unique Words per Category (Examples)")

for category, words in unique_words_by_category.items():
    st.write(f"✅ **{len(words)} unique words** in category: **{category}**")
    st.write("🔹 Example words:", words[:20])


# app_top_words.py
import streamlit as st
import pandas as pd
import os
from collections import Counter
import plotly.express as px
import matplotlib.font_manager as fm

st.title("Bangla News: Top Words per Category")

# ---------------------------
# Use df_balanced from previous steps
# ---------------------------
st.subheader("Tokenized Dataset Sample")
st.dataframe(df_balanced[['category', 'token_list']].head(5))

# ---------------------------
# Compute top 100 words per category
# ---------------------------
category_top_words = {}

for category in df_balanced['category'].unique():
    tokens = df_balanced[df_balanced['category'] == category]['token_list']
    all_tokens = [token for sublist in tokens for token in sublist]
    word_freq = Counter(all_tokens)
    category_top_words[category] = word_freq.most_common(100)

# ---------------------------
# Create top_words_df
# ---------------------------
rows = []
for category, words in category_top_words.items():
    for word, freq in words:
        rows.append({'category': category, 'word': word, 'frequency': freq})

top_words_df = pd.DataFrame(rows)

st.subheader("Top Words DataFrame Sample")
st.dataframe(top_words_df.head(10))

# ---------------------------
# Bar Chart for Top 15 Words per Category
# ---------------------------
font_path = "/Users/ronjonkar/Desktop/Streamlit/NotoSansBengali-Regular.ttf"  # update if needed
fallback_font_family = "Noto Sans Bengali"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_family = fallback_font_family
else:
    st.warning("Font file not found. Using fallback font.")
    font_family = "Nikosh"

st.subheader("Top 15 Words per Category")

for category in top_words_df['category'].unique():
    cat_df = (
        top_words_df[top_words_df['category'] == category]
        .sort_values(by='frequency', ascending=False)
        .head(15)
        .sort_values(by='frequency', ascending=True)  # horizontal plot
    )

    fig = px.bar(
        cat_df,
        x='frequency',
        y='word',
        orientation='h',
        title=f"শ্রেণী: {category} - শীর্ষ ১৫ শব্দ",
        labels={'frequency': 'Frequency', 'word': 'Word'},
        text='frequency',
        color='frequency',
        color_continuous_scale='Viridis'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        font=dict(family=font_family, size=16),
        coloraxis_colorbar=dict(title="ঘনত্ব")
    )

    st.plotly_chart(fig, use_container_width=True)


# app_bigram.py
import streamlit as st
import pandas as pd
from collections import Counter
from itertools import tee
import plotly.express as px
import os
import matplotlib.font_manager as fm

st.title("Bangla News: Top Bigram Words per Category")

# ---------------------------
# Use df_balanced from previous steps
# ---------------------------
st.subheader("Tokenized Dataset Sample")
st.dataframe(df_balanced[['category', 'token_list']].head(5))

# ---------------------------
# Generate Bigrams
# ---------------------------
def generate_bigrams(tokens):
    a, b = tee(tokens)
    next(b, None)
    return zip(a, b)

category_bigram_freq = {}

for category in df_balanced['category'].unique():
    tokens = df_balanced[df_balanced['category'] == category]['token_list']

    all_bigrams = []
    for token_list in tokens:
        bigrams = generate_bigrams(token_list)
        all_bigrams.extend([' '.join(pair) for pair in bigrams])

    # Count bigrams
    bigram_counter = Counter(all_bigrams)
    category_bigram_freq[category] = bigram_counter.most_common(30)

# ---------------------------
# Create DataFrame for plotting
# ---------------------------
bigram_rows = []
for category, bigrams in category_bigram_freq.items():
    for bigram, freq in bigrams:
        bigram_rows.append({
            'category': category,
            'bigram': bigram,
            'frequency': freq
        })

bigram_df = pd.DataFrame(bigram_rows)

st.subheader("Top 30 Bigrams per Category")
st.dataframe(bigram_df.head(10))

# ---------------------------
# Bar Chart for Bigram
# ---------------------------
font_path = "/Users/ronjonkar/Desktop/Streamlit/NotoSansBengali-Regular.ttf"  # adjust for your system
fallback_font_family = "Noto Sans Bengali"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_family = fallback_font_family
else:
    st.warning("Font file not found. Using fallback font.")
    font_family = "Nikosh"

st.subheader("Top 15 Bigrams Bar Charts per Category")

for category in bigram_df['category'].unique():
    cat_df = (
        bigram_df[bigram_df['category'] == category]
        .sort_values(by='frequency', ascending=False)
        .head(15)
        .sort_values(by='frequency', ascending=True)  # horizontal plot
    )

    fig = px.bar(
        cat_df,
        x='frequency',
        y='bigram',
        orientation='h',
        title=f"শ্রেণী: {category} - শীর্ষ ১৫ বাইগ্রাম",
        labels={'frequency': 'Frequency', 'bigram': 'Bigram'},
        text='frequency',
        color='frequency',
        color_continuous_scale='Viridis'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        font=dict(family=font_family, size=16),
        coloraxis_colorbar=dict(title="ঘনত্ব")
    )

    st.plotly_chart(fig, use_container_width=True)

# app_wordcloud.py
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
from collections import Counter

st.title("Bangla News: WordCloud for Unigrams")

# ---------------------------
# Use df_balanced from previous steps
# ---------------------------
st.subheader("Tokenized Dataset Sample")
st.dataframe(df_balanced[['category', 'token_list']].head(5))

# ✅ Bangla font path (adjust for your system)
font_path = "/Users/ronjonkar/Desktop/Streamlit/NotoSansBengali-Regular.ttf"
if not os.path.exists(font_path):
    st.warning("Bangla font file not found. WordCloud may not render properly.")
    font_path = None  # WordCloud will use default font

# ---------------------------
# Compute top words per category if not already available
# ---------------------------
category_top_words = {}
for category in df_balanced['category'].unique():
    tokens = df_balanced[df_balanced['category'] == category]['token_list']
    all_tokens = [token for sublist in tokens for token in sublist]
    word_freq = Counter(all_tokens)
    category_top_words[category] = word_freq.most_common(100)

# ---------------------------
# Select category to display
# ---------------------------
selected_category = st.selectbox("Select Category", list(category_top_words.keys()))

# Convert list of tuples to dict for WordCloud
freq_dict = dict(category_top_words[selected_category])

# ---------------------------
# Generate WordCloud
# ---------------------------
wc = WordCloud(
    font_path=font_path,
    width=1000,
    height=500,
    background_color='white',
    max_words=100
).generate_from_frequencies(freq_dict)

# ---------------------------
# Plot WordCloud using matplotlib
# ---------------------------
fig, ax = plt.subplots(figsize=(12, 6))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
ax.set_title(f"{selected_category} category - Top 100 Words", fontsize=16)

st.pyplot(fig)


# app_temporal.py
import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.title("Bangla News Temporal Insights")

# ------------------ Bangla date parsing ------------------
BN_MONTHS = {
    "জানুয়ারি": 1, "ফেব্রুয়ারি": 2, "মার্চ": 3, "এপ্রিল": 4, "মে": 5, "জুন": 6,
    "জুলাই": 7, "আগস্ট": 8, "সেপ্টেম্বর": 9, "অক্টোবর": 10, "নভেম্বর": 11, "ডিসেম্বর": 12
}
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def parse_bangla_date(date_str: str):
    if not isinstance(date_str, str):
        return None
    cleaned = re.sub(r"^[^\d\u09E6-\u09EF]+", "", date_str.strip())
    cleaned = cleaned.translate(BN_DIGITS)
    pattern1 = re.compile(r"(?P<day>\d{1,2})\s+(?P<month>[^\s]+)\s+(?P<year>\d{4})(?:,\s*(?P<time>\d{1,2}:\d{2}))?")
    pattern2 = re.compile(r"(?P<time>\d{1,2}:\d{2}),\s*(?P<day>\d{1,2})\s+(?P<month>[^\s]+)\s+(?P<year>\d{4})")
    match = pattern1.search(cleaned) or pattern2.search(cleaned)
    if not match: return None
    gd = match.groupdict()
    day = int(gd["day"])
    month = BN_MONTHS.get(gd["month"], None)
    year = int(gd["year"])
    if month is None: return None
    time_part = gd.get("time") or "00:00"
    try:
        return datetime.strptime(f"{day:02d}-{month:02d}-{year} {time_part}", "%d-%m-%Y %H:%M")
    except ValueError:
        return None

# ---------------------------
# Parse Bangla dates
# ---------------------------
if 'datetime' not in df.columns:
    df['datetime'] = df['published_date'].apply(parse_bangla_date)

# ---------------------------
# Unparsed / Parsed rows
# ---------------------------
unparsed = df[df["datetime"].isna()]["published_date"].unique()
st.write(len(unparsed), "unique unparsed formats")
st.write(unparsed[:50])  # show first 50

df_parsed = df[df['datetime'].notna()].copy()
df_parsed.reset_index(drop=True, inplace=True)
st.write(f"Working with {len(df_parsed)} rows (parsed successfully).")

df_unparsed = df[df['datetime'].isna()].copy()
st.write(f"Unparsed rows are {len(df_unparsed)} (ignored for now).")

# ---------------------------
# Extract temporal features
# ---------------------------
df_parsed['year'] = df_parsed['datetime'].dt.year
df_parsed['month'] = df_parsed['datetime'].dt.month
df_parsed['day'] = df_parsed['datetime'].dt.day
df_parsed['hour'] = df_parsed['datetime'].dt.hour
df_parsed['weekday'] = df_parsed['datetime'].dt.day_name()

# ---------------------------
# 1️⃣ Number of articles per year
# ---------------------------
year_counts = df_parsed['year'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=year_counts.index, y=year_counts.values, palette="viridis", ax=ax)
ax.set_title("Articles per Year")
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------------------
# 2️⃣ Articles per month (overall)
# ---------------------------
month_counts = df_parsed['month'].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=month_counts.index, y=month_counts.values, palette="magma", ax=ax)
ax.set_title("Articles per Month")
ax.set_ylabel("Count")
st.pyplot(fig)

# ---------------------------
# 3️⃣ Articles per weekday
# ---------------------------
weekday_counts = df_parsed['weekday'].value_counts().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
)
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette="coolwarm", ax=ax)
ax.set_title("Articles per Weekday")
ax.set_ylabel("Count")
plt.xticks(rotation=45)
st.pyplot(fig)

# ---------------------------
# 4️⃣ Articles per month per year (heatmap)
# ---------------------------
monthly_year_counts = df_parsed.groupby(['year','month']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12,6))
sns.heatmap(monthly_year_counts, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
ax.set_title("Articles per Month per Year")
ax.set_ylabel("Year")
ax.set_xlabel("Month")
st.pyplot(fig)

# ---------------------------
# Category-wise analysis
# ---------------------------
if 'category' in df_parsed.columns:
    # 1️⃣ Articles per year per category
    year_cat_counts = df_parsed.groupby(['category','year']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12,6))
    for cat in year_cat_counts.index:
        ax.plot(year_cat_counts.columns, year_cat_counts.loc[cat], marker='o', label=cat)
    ax.set_title("Articles per Year by Category")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Articles")
    ax.legend()
    st.pyplot(fig)

    # 2️⃣ Articles per month per category (all years)
    month_cat_counts = df_parsed.groupby(['category','month']).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(month_cat_counts, cmap="YlOrRd", annot=True, fmt="d", ax=ax)
    ax.set_title("Monthly Article Distribution by Category (All Years)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Category")
    st.pyplot(fig)

    # 3️⃣ Articles per weekday per category
    weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    weekday_cat_counts = df_parsed.groupby(['category','weekday']).size().unstack(fill_value=0)[weekday_order]
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(weekday_cat_counts, cmap="coolwarm", annot=True, fmt="d", ax=ax)
    ax.set_title("Weekday Article Distribution by Category")
    ax.set_xlabel("Weekday")
    ax.set_ylabel("Category")
    st.pyplot(fig)

# app_embeddings.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

st.title("Bangla News: BERT Embeddings")

# ---------------------------
# Use balanced dataset
# ---------------------------
texts = df_balanced['cleaned_content'].tolist()
st.write(f"Total articles to encode (balanced dataset): {len(texts)}")

# ---------------------------
# Load Bangla BERT model (cached)
# ---------------------------
@st.cache_resource
def load_model(model_name="sagorsarker/bangla-bert-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# Determine device
# ---------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Mac GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

st.write(f"Using device: {device}")

# ---------------------------
# Embedding function
# ---------------------------
def get_embeddings(text_list, batch_size=32, device='cpu'):
    all_embeddings = []

    model.to(device)
    for i in tqdm(range(0, len(text_list), batch_size), desc="Encoding Batches"):
        batch_texts = text_list[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True,
                           return_tensors="pt", max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(batch_embeddings)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_embeddings

# ---------------------------
# Compute embeddings with progress bar
# ---------------------------
if st.button("Compute BERT Embeddings"):
    with st.spinner("Computing embeddings for balanced dataset... ⏳"):
        embeddings = get_embeddings(texts, batch_size=64, device=device)
    st.success("Embeddings computed ✅")
    st.write("Embeddings shape:", embeddings.shape)
