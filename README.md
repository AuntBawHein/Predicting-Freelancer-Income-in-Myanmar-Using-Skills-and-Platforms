# Predicting-Freelancer-Income-in-Myanmar-Using-Skills-and-Platforms
To help freelancers in Myanmar predict income based on skills and platforms using Python.

1. Introduction

Since **COVID-19** and the recent **political situation** in Myanmar, many young people have started working online.  
A lot of them use platforms like **Facebook**, **Telegram**, **Fiverr**, and **Upwork** to find freelance jobs — even if they live in small towns or villages.

In this project, I want to:
- Understand what factors affect **how much money freelancers earn**
- Try using **machine learning** to **predict freelancer income**
- Work with data that’s similar to what real freelancers in Myanmar might have in **2025**

---

2. Goal of This Project

My main goals are to:
- Find out what things make freelancers earn **more or less money online**
- Build a machine learning model (a **regression model**) to **predict their monthly income in USD**
- Share useful results with:
  - Freelancers  
  - Teachers and training programs  
  - Websites or platforms that support online work

---

3. Why I Chose This Project

I chose this project because:

- A lot of people in Myanmar are starting freelance jobs online
- The income is **very different** — some people earn a lot, others very little  
- I want to understand **why that happens**
- This project can help:
  - People choose better freelance jobs  
  - Teachers create smarter training  
  - Platforms grow the **online job market in Myanmar**

---

4. Tools and Libraries I Used in Python

Here are the Python tools I used for this project:

1. **Pandas** – to read and explore the data  
2. **NumPy** – to do calculations and work with numbers  
3. **Matplotlib** – to create simple graphs  
4. **Seaborn** – to make more detailed and colorful charts  
5. **Scikit-learn** – to train and test machine learning models like RandomForest  
6. **TensorFlow** – I used this in Step 7 to test a neural network model

---

Step 1: Define Scope / Objectives of the Project

In this step, I will explain **what my project is about** and **what I plan to do**.

---

Objective

- I want to understand what factors affect **how much money freelancers in Myanmar earn online**.
- I will use **machine learning** to predict a freelancer’s **monthly income** based on their skills, background, and work activity.

---

Scope

- My project focuses on data from **freelancers living in Myanmar**.
- I will try to predict just **one thing**: how much money they earn each month (in **USD**).
- To make this prediction, I will use information such as:
  - Age  
  - Gender  
  - Primary and secondary skills  
  - Hours worked per week  
  - Internet quality  
  - Type of device used  
  - Platform (like Upwork or Fiverr)

---

Step 2: Load and Look at the Data

In this step, I opened my dataset — the file that contains **all the freelancer information**.  

I used Python to check what’s inside the file, like:

- The **column names** (what kind of info each row has)
- The **type of data** (numbers, categories, text)
- And whether the data is **clean or duplicate**

This is a very important step, because before I do any machine learning, I need to understand what kind of data I’m working with.

---

What is Pandas?

Pandas is a Python tool that helps me work with data in **tables**.

---

Why do I use Pandas?

I use Pandas because it makes it really easy to:

- **Open** the file and look at it
- **Remove repeated rows** (duplicates)
- **Find missing values**
- **See patterns** quickly
- Work with data in a clean and structured way

Basically, Pandas helps me **understand and fix the data** before training my model.

---

What is NumPy?

NumPy is another Python tool that helps me do **math with data**.

You can think of it like a **super calculator** that works with **big lists of numbers** really fast.

---

Why do I use NumPy?

I use NumPy when I want to:

- **Calculate averages** (like the average freelancer income)
- **Count things** (like how many people have stable internet)
- **Do math operations** with lots of numbers at once

Together, **Pandas and NumPy** helped me explore and prepare my data the right way.

---

Code 

import pandas as pd  # I use pandas to work with table data

# Load the dataset
df = pd.read_csv("myanmar_freelancer_income_2025.csv")

# Show the first few rows to see what's inside
print(df.head())

# Check column names, data types, and missing values
df.info()

# Show summary statistics for number columns
print(df.describe())

Output 

First 5 rows of the data:
                          freelancer_id  age  gender        city  \
0  495f34dc-c37f-4f50-9c98-41536408cd84   46  Female  Mawlamyine   
1  95428dcc-6d11-4faf-9999-6808429d77f9   32    Male      Sittwe   
2  196eba5a-b2a6-41fd-a1ee-e4113694448b   25    Male    Taunggyi   
3  3d2bf5fd-aed2-4ede-878f-ff221bdf8267   38   Other     Pathein   
4  1807a7bc-c78c-4ec3-9bae-ebbf8b8c11a6   36    Male    Taunggyi   

  education_level    primary_skill    secondary_skill  platform  \
0             NaN   Graphic Design  Digital Marketing  Telegram   
1      University   Graphic Design         Data Entry  Facebook   
2      University      Programming     Graphic Design    Fiverr   
3      University      Translation        Programming    Fiverr   
4      University  Web Development  Digital Marketing  Facebook   

   hours_per_week  experience_years internet_stability has_smartphone_only  \
0               6               9.6               Fair                  No   
1              43               0.9               Good                  No   
2              37               7.7               Poor                  No   
3              28               5.4               Fair                 Yes   
4              16               0.6               Good                 Yes   

     client_type  monthly_income_usd  
0          Local              357.36  
1  International              383.84  
...
25%      27.000000       19.000000          2.400000          261.612500
50%      36.000000       32.000000          4.900000          355.385000
75%      45.000000       46.000000          7.500000          465.127500
max      54.000000       59.000000         10.000000          930.350000

Conclusion for Step 2: Load and Look at the Data

When I opened the dataset, I found it had **7,000 rows** (freelancers) and **14 columns** (different details like skills, income, and hours).  
Some columns were missing information — like the **education level** and **secondary skill**. For example, only about **6,317 people filled in their education**, so not everyone gave that info.  
The **highest income** in the data was **$930.35**, and the **lowest was $50**, which shows that some freelancers earn way more than others.  
I also learned that most freelancers work around **32 hours per week**, but some work **as little as 5 hours**, while others work **almost 60 hours**.  
This step helped me understand what’s inside the data — and what I might need to clean or fix next.

Step 3: Clean the Data

What I’m going to do:

In this step, I’ll **check the data for problems** — like missing info or strange numbers — and then clean it up.  
If the data is messy, my machine learning model might get confused or give wrong answers, so this step is **super important**.

---

Tools I’ll Use:

I’ll use the **Pandas** library to check and fix the data.

Here are some of the tools I’ll use:

- `isnull()` – to check for missing (empty) data  
- `dropna()` – to **delete rows** that are missing too much info  
- `fillna()` – to **fill in missing values** using an average or most common value  
- `describe()` – to check if the numbers look normal  
- `boxplot()` – to **find strange or extreme numbers** using a chart

---

Steps I’ll Follow:

1. **Check for missing values**
   
   First, I’ll look for blanks — like when someone didn’t write their age, education, or skill.  
   These are missing values that need to be fixed.

2. **Fix or remove the missing info**
   
   - If there are just **a few missing rows**, I’ll delete them.  
   - If there are **a lot of missing values**, I’ll fill them with something that makes sense — like the **average** or the **most common answer**.

3. **Find strange or impossible values**
   
   For example, if someone wrote that they work **200 hours a week** or earn **$1,000,000**, that’s probably not true.  
   I’ll use `describe()` and draw a `boxplot()` to spot these outliers.

4. **Check the data types**  
   I’ll make sure that numbers (like income or age) are stored as numbers — not as text.  
   If anything looks wrong, I’ll fix it.

---

What is Pandas?  

Pandas is like **Excel for Python**. It helps me open data files, look at them, and fix uncessary data.  
I can use Pandas to sort, clean, and prepare my data before building the model.

---

Why do I need Pandas in this step?  

Because Pandas helps me **find problems in the data quickly** — like missing info, wrong types, or duplicates.  
It also helps me clean everything up so the model can learn better.

---

What is Seaborn?  

Seaborn is a tool that helps me **draw nice charts** in Python.  
It works with Pandas and helps me **see patterns** or **find strange values**.

---

Why use Seaborn in this step?

I use Seaborn to make **boxplots**, which help me find strange numbers easily.  
It’s much easier to **see outliers in a chart** than by looking at numbers one by one.

---

Code 

import pandas as pd                # I use pandas to open and work with my data table
import seaborn as sns              # I use seaborn to make charts (like boxplots)
import matplotlib.pyplot as plt    # I use matplotlib to display the charts on screen

# I load the dataset from a CSV file into a table called df.
df = pd.read_csv("myanmar_freelancer_income_2025.csv")

# I fill in missing values:
# If someone didn’t write their education level, I write "Unknown".
# If they didn’t list a secondary skill, I write "None".
df.fillna({'education_level': 'Unknown', 'secondary_skill': 'None'}, inplace=True)

# I print out summary statistics — like average, min, and max — for all number columns
# This helps me see if any values are too high or too low
print(df.describe())

# I draw a boxplot for the monthly income to check for very high or strange values
# This helps me find outliers (people who earn way more than most others)
sns.boxplot(x=df['monthly_income_usd'])
plt.show()

# I find the freelancers who earn more than $800 — these are considered high earners
outliers = df[df['monthly_income_usd'] > 800]

# I print their freelancer ID and how much they earn
print(outliers[['freelancer_id', 'monthly_income_usd']])

# I count how many people earn more than $800
print("Total freelancers earning over $800:", outliers.shape[0])

Output 

Missing values in each column:
freelancer_id            0
age                      0
gender                   0
city                     0
education_level        683
primary_skill            0
secondary_skill        897
platform                 0
hours_per_week           0
experience_years         0
internet_stability       0
has_smartphone_only      0
client_type              0
monthly_income_usd       0
dtype: int64

Summary of numeric columns:
               age  hours_per_week  experience_years  monthly_income_usd
count  7000.000000     7000.000000       7000.000000         7000.000000
mean     36.128857       31.986286          4.961457          367.693051
std      10.617523       15.857509          2.899511          150.417005
min      18.000000        5.000000          0.000000           50.000000
25%      27.000000       19.000000          2.400000          261.612500
50%      36.000000       32.000000          4.900000          355.385000
75%      45.000000       46.000000          7.500000          465.127500
max      54.000000       59.000000         10.000000          930.350000

Boxplot for monthly income:


Data types of each column:
freelancer_id           object
age                      int64
gender                  object
city                    object
education_level         object
primary_skill           object
secondary_skill         object
platform                object
hours_per_week           int64
experience_years       float64
internet_stability      object
has_smartphone_only     object
client_type             object
monthly_income_usd     float64
dtype: object

People who earn more than $800:
                             freelancer_id  monthly_income_usd
5161  cc83bdc2-7da4-4366-b739-c854d8c1d825              930.35
726   11cea66e-fdc6-420f-8adc-ea9de2107a9b              923.51
6839  febed89a-7d55-4a15-9bfe-dbf4020269e3              887.93
5695  a913c2ba-e036-4429-9dfe-1b9c29b7b7c5              883.03
1529  748ab940-5f93-4801-9ced-e48ff49d7791              880.59
5949  3492fb87-9076-4d45-bf48-be10ca4319c7              879.31
2902  d6b535fe-a2c6-40e7-b570-4e67e9c2dc96              856.80
5209  a0699bca-fd27-46a0-9ec2-54a55a850c39              856.65
3797  17bd51c7-0580-4820-b72e-a4b5af0fbe8f              853.91
320   e6ae99a3-2299-4e4a-956b-cf20efeaaa0b              847.69
4901  a94f27fc-8583-484e-be2f-73a02ee3eced              844.57
4106  359a5c11-b5d7-4039-bf60-dd81eb8722b2              842.58
6489  d1d7d33f-a6dc-4623-a071-99f4e3a5159c              841.54
2645  85fbfa4b-14bb-4afe-867a-0a399409eb5e              838.23
3136  e9105e94-b0ce-446a-badc-430aba8b07c8              836.40
3062  25f8ef2d-7d7d-4d05-afc9-c6c14ff960f6              832.19
2986  f8d64621-7490-4eef-8753-c91551304909              829.72
568   f3453a5b-4c14-4c40-9c4b-8ea8c9a19a1f              827.57
2723  bcf517f1-f6d5-47e7-80c1-f5f80c8e6ae3              826.60
6259  4094656c-344e-4643-ac25-30c1d59c5f4d              826.03
5716  ea4fc52c-06b8-4287-a469-3ed97f8d2b6c              816.80
1026  8adf8dd6-0b3f-489f-b9eb-9e9338cf9c55              815.40
6234  b9268c1e-42f9-4019-b971-acf6600b779d              813.84
3092  a64d2cb3-c531-4685-a74e-d98fd236d76d              813.18
5074  8f2a6d12-88c4-4139-af34-5252ca826480              812.28
6367  7c9a59fb-0e4e-49de-b985-bf9735297b43              811.25
983   8ad44325-1c6a-4907-a4f7-9ca21924da87              811.09
123   7e663ca6-6892-45ab-b8e8-52c8ad2a9902              806.73
5485  a1804f29-9278-4fc0-8a95-4a6a39d9898f              805.49
4466  1fefb799-8d94-4d2c-bbde-21b4ca3c172e              803.05
6290  c4e758ca-125a-46d6-9642-ce4e6f2673fe              801.12

Total freelancers earning over $800: 31

![image](https://github.com/user-attachments/assets/7dad7765-9880-479a-9ff2-ac97316921a1)

Conclusion for Step 3: Clean the Data

After cleaning the data, I found that **most freelancers in Myanmar earn around $355 per month**.  
From the boxplot, I saw that **most incomes are between $260 and $465**, which is the normal range.  
But I also found **31 freelancers who earn more than $800** — and one even earns up to **$930**.  
These are called **outliers** because they make a lot more money than the others.  
I also cleaned the dataset by **filling in missing education and skill data**, so now the data is complete and ready for machine learning.

---

Step 4: Explore the Data (EDA – Exploratory Data Analysis)

What I’m going to do in this step:

- I’ll **look at the data using graphs and charts** instead of just numbers.
- I want to see if I can **spot patterns** — like who earns more based on skill, platform, or hours worked.
- My goal is to understand **what affects a freelancer’s income** before building a prediction model.

---

Tools I’ll Use:

1. **Pandas** – to group and prepare the data  
2. **Matplotlib** – to show the charts  
3. **Seaborn** – to make beautiful and clear graphs  

---

What is Pandas, and why do I use it?

**Pandas is like the Excel of Python.**  

I use it to:

- Open and clean my data
- Group it by skills, platforms, or cities
- Calculate things like average income per group

It helps me **prepare the data before I draw any charts**.

---

What is Matplotlib, and why do I use it?

**Matplotlib is the basic tool for showing graphs in Python.**  
Even if I use other libraries (like Seaborn), Matplotlib is the one that actually **displays the chart** on the screen.

I use it when I want to:

- Show bar graphs or line charts
- Customize how the graph looks

---

What is Seaborn, and why do I use it?

**Seaborn makes prettier and smarter charts.**  

It works together with Pandas and Matplotlib to:

- Compare income across platforms
- Show trends like “more hours = more money”
- Find patterns in categories like skill, gender, or education

It helps me **see the story behind the numbers**.

---

Code 

# I import pandas so I can work with data in table format.
import pandas as pd

# I import seaborn to draw nice, clear bar charts and visualizations.
import seaborn as sns

# I import matplotlib to help me display the chart on the screen.
import matplotlib.pyplot as plt

# I load the freelancer data from the CSV file.
df = pd.read_csv("myanmar_freelancer_income_2025.csv")

# I fill in missing values so they don't cause problems later.
# For example, if someone didn't write their education level, I put "Unknown".
# If they didn’t have a secondary skill, I put "None".
df.fillna({'education_level': 'Unknown', 'secondary_skill': 'None'}, inplace=True)

# I group the data by platform (like Upwork, Fiverr, etc.)
# and calculate the average monthly income for each platform.
platform_income = df.groupby('platform')['monthly_income_usd'].mean().sort_values(ascending=False)

# I turn the result into a new table so it’s easier to use for plotting.
platform_income_df = platform_income.reset_index()

# I make a bar chart to show which platforms pay more or less.
plt.figure(figsize=(8, 5))  # I make the chart a bit wider so everything fits nicely.

# I draw a bar chart using seaborn.
# The x-axis is the name of the platform (like Upwork), and the y-axis is average income.
sns.barplot(data=platform_income_df, x='platform', y='monthly_income_usd', order=platform_income_df['platform'])

# I give the chart a title so the audience knows what it shows.
plt.title("Average Monthly Income by Freelancing Platform")

# I label the y-axis to show this is about how much money people earn.
plt.ylabel("Income (USD)")

# I label the x-axis to show this is about the platform names.
plt.xlabel("Platform")

# I rotate the platform names so they’re easier to read.
plt.xticks(rotation=45)

# I make sure the whole chart fits nicely on the screen.
plt.tight_layout()

# I show the chart on the screen.
plt.show()

# I print out the exact average income for each platform.
# This lets me see the actual numbers clearly in the terminal/output.
print(platform_income_df)

Output 

Average monthly income on each platform:
   platform  monthly_income_usd
0    Fiverr          368.907244
1    Upwork          367.966127
2  Facebook          367.898241
3  Telegram          366.500594
4    Others          366.370941

![image](https://github.com/user-attachments/assets/b78197a1-1e6f-4dde-8731-ee8b6998f558)

Conclusion for Step 4: Explore the Data

The result I found is that **Fiverr gives the highest average income**, around **$368.91 per month**.  
I compared this to other platforms like **Upwork** ($367.96) and **Facebook** ($367.89), and the difference is **small**, but Fiverr is still a little higher.  
This tells me that even though the platforms are very close, **which platform you choose can slightly affect your income**.  
All the platforms give similar results, but Fiverr is still at the top in my data.  
So I learned that platform choice **might give you a small boost**, even if everything else stays the same.

---

Step 5: **Turn Words into Numbers (Feature Engineering)**

---

What am I doing in this step?

In this step, I’m getting my data ready for the machine learning model.  
That means I need to **turn all the words into numbers**, because:

- The model can only understand **numbers**, not words.
- If I leave words like “Yangon,” “Graphic Design,” or “Facebook,” the model will get confused.
- This process is called **feature engineering** — it means I’m shaping the data in a way the model can understand and learn from.

---

Tools I’ll Use:
- **Pandas** – for cleaning my data  
- **LabelEncoder** – to change simple words into numbers  
- **OneHotEncoder** – to turn choices (like platforms or skills) into multiple columns  
- **ColumnTransformer** – to apply these changes to just the right parts of my data

---

1. What is Pandas?

Pandas is like a tool that helps me **work with tables in Python**.
I use it to **open, clean, and prepare** the data before changing words into numbers.

---

2. Why do I need Pandas for this step?

Because I need to **pick which columns I want to change**, and Pandas makes that super easy.  
It helps me keep the data clean and ready for encoding.

---

3. What is LabelEncoder?

LabelEncoder is a tool that turns simple words into simple numbers.  
For example:
- “Male” → 0  
- “Female” → 1  
- “Other” → 2

---

4. Why do I use LabelEncoder?

Because **some columns only have one answer per person** (like gender), and LabelEncoder works well for that.  
It gives each word a number the model can understand.

---

5. What is OneHotEncoder?

OneHotEncoder turns one column into **many columns**, using 0s and 1s.  
For example, if someone uses **Upwork**, **Fiverr**, or **Facebook**, it becomes:

| Upwork | Fiverr | Facebook |  
|--------|--------|----------|  
|   1    |   0    |    0     ← This person uses Upwork  

---

6. Why do I use OneHotEncoder?

Because **some columns have lots of possible answers**, like skills or platforms.  
The model can’t guess what those words mean, so I split them into yes/no columns.

---

7. What is ColumnTransformer?

ColumnTransformer helps me **only change the columns I need**.  
For example: I want to change “platform” and “primary skill” but **leave the rest alone**.

---

8. Why do I use ColumnTransformer?

Because it saves me time and keeps my code clean.  
It lets me tell Python: **“Only change these columns, and skip the others.”**

---

Code 

# I import the tools I need to change words into numbers
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# I open the freelancer dataset and load it into a table
df = pd.read_csv("myanmar_freelancer_income_2025.csv")

# I clean the data by filling in missing values
# For example, if someone didn’t write their education, I put "Unknown"
# If they didn’t write their second skill, I put "None"
df.fillna({'education_level': 'Unknown', 'secondary_skill': 'None'}, inplace=True)

# I make a copy of the original table so I can safely make changes without losing the real one
df_encoded = df.copy()

# I decide which columns have simple categories that I can turn into numbers using LabelEncoder
label_columns = ["gender", "education_level", "internet_stability", "has_smartphone_only"]

# I go through each of these label columns and turn text into numbers
for col in label_columns:
    le = LabelEncoder()                     # I create a tool that changes words into numbers
    df_encoded[col] = le.fit_transform(df_encoded[col])  # I apply it to the column

# Some columns have too many different words, so I use OneHotEncoder for them
# For example, platform = Upwork, Fiverr, Facebook, etc. — I turn that into 0s and 1s
one_hot_columns = ["city", "platform", "primary_skill", "client_type"]

# I use ColumnTransformer to apply OneHotEncoder only to those columns, not the whole table
ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False, drop='first'), one_hot_columns)
    ],
    remainder='passthrough'  # I keep all other columns (like age, hours, etc.) the same
)

# I apply the transformation — now the whole dataset has numbers only
df_encoded_final = ct.fit_transform(df_encoded)

# I get the new column names created by OneHotEncoder (like city_Yangon, platform_Upwork, etc.)
encoded_col_names = ct.named_transformers_['onehot'].get_feature_names_out(one_hot_columns)

# I collect all the column names: the new ones and the ones that stayed the same
all_col_names = list(encoded_col_names) + [col for col in df_encoded.columns if col not in one_hot_columns]

# I turn the final result into a clean DataFrame (table) and make sure all data types are correct
df_final = pd.DataFrame(df_encoded_final, columns=all_col_names).convert_dtypes()

# I save my final table as a CSV file so I can use it in the next step (model training)
df_final.to_csv("df_final_encoded.csv", index=False)


What I’m going to do:

[myanmar_freelancer_income_2025.csv](https://github.com/user-attachments/files/19716375/myanmar_freelancer_income_2025.csv)
[df_final_encoded.csv](https://github.com/user-attachments/files/19716367/df_final_encoded.csv)

Output 

	city_Mandalay	city_Mawlamyine	city_Naypyidaw	city_Pathein	city_Sittwe	city_Taunggyi	city_Yangon	platform_Fiverr	platform_Others	platform_Telegram	...	freelancer_id	age	gender	education_level	secondary_skill	hours_per_week	experience_years	internet_stability	has_smartphone_only	monthly_income_usd
0	0	1	0	0	0	0	0	0	0	1	...	495f34dc-c37f-4f50-9c98-41536408cd84	46	0	2	Digital Marketing	6	9.6	0	0	357.36
1	0	0	0	0	1	0	0	0	0	0	...	95428dcc-6d11-4faf-9999-6808429d77f9	32	1	1	Data Entry	43	0.9	1	0	383.84
2	0	0	0	0	0	1	0	1	0	0	...	196eba5a-b2a6-41fd-a1ee-e4113694448b	25	1	1	Graphic Design	37	7.7	2	0	219.43
3	0	0	0	1	0	0	0	1	0	0	...	3d2bf5fd-aed2-4ede-878f-ff221bdf8267	38	2	1	Programming	28	5.4	0	1	386.73
4	0	0	0	0	0	1	0	0	0	0	...	1807a7bc-c78c-4ec3-9bae-ebbf8b8c11a6	36	1	1	Digital Marketing	16	0.6	1	1	173.01

Conclusion for Step 5: Feature Engineering (Turning Words into Numbers)

The result I got is a **new table** where all the words were changed into **numbers**, so the computer can understand them.  
For example, instead of saying “Male” or “Female,” I turned them into numbers like **0 for Male**, **1 for Female**, and so on.  
I also turned city names like “Yangon” or “Mandalay” into separate columns — so if someone is from Yangon, the column called `city_Yangon` will show a **1** (which means “Yes”), and **0** if they’re not.  
Now, things like **“Graphic Design”** or **“Upwork”** are no longer text — they’re just clean numbers like `platform_Upwork = 1`, which makes it much easier for the model to learn.  
This new table is now **100% ready** for machine learning because everything is in **number form**, and that’s the only thing the model understands when it starts learning.

---

Step 6: Split the Data

What I’m doing in this step:

- I’m going to **divide my dataset into two parts** — one part for training the machine learning model and one part for testing it.
- This step is really important because I need to check if the model is **learning properly** and not just memorizing everything.

Easy way to understand it:

- The **training data is like classwork or practice problems** — the model uses this part to learn patterns (like what makes someone earn more or less).
- The **testing data is like a test or exam** — the model has never seen this part before, so I can check if it’s truly learned anything useful.

---

Tools I’m using:

---

1. What is Pandas?

= **Pandas is a Python tool that helps me organize and work with data in tables.**  
I use it to choose the input data (like age, skill, hours worked) and the target I want to predict (monthly income).

---

2. Why do I use Pandas in this step?
 
= I use Pandas to **separate the dataset into two parts**:  
- `X` = all the information the model will use to make predictions (like age, skill, platform)  
- `y` = the actual thing I want the model to predict — the freelancer’s income in USD

---

3. What is train_test_split?**

= **train_test_split is a tool that helps me split the data into a training part and a testing part.**  
It’s from a Python library called `sklearn`.

---

4. Why do I use train_test_split?

= I use it so the model can **practice on some data (training)** and then take a **test on brand new data (testing)**.  
This helps me know if the model can really **predict new answers** — not just repeat what it saw.

---

Code 

# I import pandas so I can work with data in a table format (like Excel but with code)
import pandas as pd

# I import train_test_split — this tool helps me divide the data into training and testing sets
from sklearn.model_selection import train_test_split

# I open the final cleaned dataset that only has numbers (from Step 5)
# This file is already ready for machine learning
df_final = pd.read_csv("df_final_encoded.csv")  

# I separate my data into two parts:
# X is the "input" — these are the details like skills, hours worked, age, etc.
X = df_final.drop(columns=["monthly_income_usd"])  

# y is the "output" — this is the thing I'm trying to predict: freelancer income in USD
y = df_final["monthly_income_usd"]  

# I split the dataset into training and testing groups using 80% for training, 20% for testing
# The model will learn from the training part and be tested on the testing part
# random_state=42 makes sure I get the same result every time I run this code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

# I print the total number of rows in my full dataset
print("All rows:", len(df_final))  

# I print how many rows are in the training set — this is what the model will learn from
print("Training set:", len(X_train))  

# I print how many rows are in the testing set — this is what I’ll use to check how well the model works
print("Testing set:", len(X_test))  

Output

Total rows in the dataset:  7000
Rows in training set: 5600
Rows in testing set:  1400

Conclusion for Step 6: Split the Data

The result I got is a total of **7,000 rows** in my dataset.  
I split the data so the model can **learn from 5,600 rows** (like practice questions) and **test on 1,400 rows** (like an exam).  
This setup helps me see if the model is truly learning or just memorizing what it saw.  
It’s like how we study for a test — we practice first, then take a test to see how well we understand the topic.  
Splitting the data like this is important because it shows if the model can make smart guesses on new, unseen data.

---

Step 7: Build the Models

What I’m doing in this step:  

- I’m building smart models that try to **guess how much money freelancers earn** each month.  
- These models will look at the data (like hours worked, platform, skills, etc.) and try to learn patterns.  
- I’ll try **three different models** to see which one makes the best guesses.

---

Tools and Models I’m Using (and Why)

---

Pandas – What is it? 

Pandas is a tool that helps me work with my data.  
I use it to duplicate and clean my table so the model can learn properly.

---

Why I use Pandas:  

I use it to **get my data ready** before building the models.  
It helps me pick what columns I want to use and split the data into inputs (X) and answers (y).

---

XGBoost – What is it?  

XGBoost is a strong machine learning model that **keeps improving by learning from its mistakes**.  
It’s known for being accurate and reliable.

---

Why I use XGBoost:  

Because it works well even if the data is big or a bit messy.  
It learns in steps and keeps getting better.

---

LightGBM – What is it?

LightGBM is a fast version of XGBoost.  
It does similar things, but it’s usually **faster when training on big data**.

---

Why I use LightGBM: 

It helps me **train the model quicker** while still giving smart results.  
This is great when I want to test things fast.

---

### **CatBoost – What is it?**  
CatBoost is really good at **understanding text data**, like city names or platforms.  
I don’t need to change the words too much — CatBoost handles that for me.

---

Why I use CatBoost:  

Because my dataset has lots of categories like “Upwork” or “Yangon.”  
CatBoost makes it easier and **saves me time** during feature engineering.

---

TensorFlow / Keras – What is it? 

These are tools I use to build something called a **neural network**, which is a model inspired by how our brain works.  
It can find deep patterns in the data.

---

Why I use TensorFlow/Keras:

If I want the model to be **more flexible and smart**, this is a great choice.  
It takes more time, but it can learn complicated things.

---

Mean Squared Error (MSE) – What is it?  

MSE is how I **check if the model made good or bad guesses**.  
It compares the real income with the predicted income and gives me a score.  
The **smaller the number, the better** the model is doing.

---



Why I use MSE:

I need a way to compare the models and pick the best one.  
MSE helps me **see which model made the closest predictions**.

---

train_test_split – What is it?

This is a tool that **splits the data into two parts** — one for training (learning) and one for testing (checking).

---

Why I use train_test_split:

So the model learns from some data and is tested on different data.  
It’s like **studying and then taking a test** — that’s how I know if the model really learned something.

---

Code 

# I import pandas so I can open and work with my data in table format
import pandas as pd

# I import a tool that helps me split my data into training and testing sets
from sklearn.model_selection import train_test_split

# I import XGBoost, a strong machine learning model that learns from mistakes
from xgboost import XGBRegressor

# I import CatBoost, a model that works well with text categories (like city names or skills)
from catboost import CatBoostRegressor

# I import mean_squared_error — this checks how far off the model's guesses are
from sklearn.metrics import mean_squared_error

# I import TensorFlow and Keras to build a deep learning model (neural network)
from tensorflow import keras

# I import layers so I can design how many parts (or layers) my neural network has
from tensorflow.keras import layers


# Step 1: Load the final dataset (all features are numeric now)
df_final = pd.read_csv("df_final_encoded.csv")  # I open the final cleaned dataset

# Step 2: Select input features (X) and target label (y)
X = df_final.drop(columns=["monthly_income_usd", "freelancer_id", "secondary_skill"])  # I remove columns that are not needed for prediction
y = df_final["monthly_income_usd"]  # This is what I want the model to predict

# Step 3: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # I split the data for learning and testing

# Step 4: Train the first model using XGBoost
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.05, random_state=42)  # I create the XGBoost model
xgb_model.fit(X_train, y_train)  # I train it with the training data
xgb_preds = xgb_model.predict(X_test)  # I use it to predict income on test data
xgb_mse = mean_squared_error(y_test, xgb_preds)  # I calculate prediction error for XGBoost

# Step 5: Train the second model using CatBoost
cat_model = CatBoostRegressor(verbose=0, random_state=42)  # I create the CatBoost model
cat_model.fit(X_train, y_train)  # I train it with the same training data
cat_preds = cat_model.predict(X_test)  # I use it to predict test data
cat_mse = mean_squared_error(y_test, cat_preds)  # I calculate prediction error for CatBoost

# Step 6: Train the third model using Neural Network (TensorFlow/Keras)
model = keras.Sequential([  # I build a simple neural network with 2 hidden layers
    layers.Dense(128, activation='relu', input_shape=[X_train.shape[1]]),  # First layer
    layers.Dense(64, activation='relu'),  # Second layer
    layers.Dense(1)  # Final output layer (for predicted income)
])
model.compile(optimizer='adam', loss='mse')  # I tell the model how to learn (optimizer + error measure)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)  # I train it for 50 rounds (epochs)
nn_preds = model.predict(X_test).flatten()  # I get predictions and make them 1D
nn_mse = mean_squared_error(y_test, nn_preds)  # I calculate prediction error for neural network

# Step 7: Print all MSE results
print("XGBoost MSE:", round(xgb_mse, 2))  # Lower MSE = better performance
print("CatBoost MSE:", round(cat_mse, 2))
print("Neural Net MSE:", round(nn_mse, 2))

# Step 8: Identify and print the best model
best_mse = min(xgb_mse, cat_mse, nn_mse)  # I find the smallest MSE
if best_mse == xgb_mse:
    print(" Best Model: XGBoost")
elif best_mse == cat_mse:
    print(" Best Model: CatBoost")
else:
    print(" Best Model: Neural Network")

Output

  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
44/44 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step
XGBoost MSE:  8822.67
CatBoost MSE: 8640.85
Neural Net MSE: 8792.02
Best Model : CatBoost
Best Model MSE:  8640.85

Conclusion for Step 7: Build the Models

The result I got is that **CatBoost was the best model** at predicting freelancer income.  
I tried three different models: **XGBoost**, **CatBoost**, and a **Neural Network**. After testing them, CatBoost had the **smallest error score — 8640.85**, which means it was the most accurate.  
For comparison, XGBoost had a higher error (**8822.67**) and the Neural Network was in the middle with **8792.02**.  
Even though all the models used the same data to learn, **CatBoost understood the patterns better** and made better guesses.  
That’s why I will choose CatBoost to continue the rest of my project — it gave me the best results.

---

Step 8: Test the Models (Evaluation)

**I’m going to:**

- Test my model to see how close its predictions are to the real incomes in the data.  
- I want to know: **Did my model guess the income well, or was it way off?**  
- This is a really important step, because a model that looks smart during training might still make bad guesses on new data.  
- So now, I’ll check if my model actually works **in real situations** — not just during practice!

---

1. What is `sklearn.metrics`?

= `sklearn.metrics` is a tool in Python that helps me **check how good or bad my model is**.  
It tells me **how close the model’s predictions are** to the real income values.

---

2. Why do I need it for this step?
   
= I use it to **measure my model’s error** — how far off its guesses were.  
For example, I can use something called **Mean Squared Error (MSE)** to see how wrong the model was.

---

3. What is Pandas?
   
= Pandas is a tool that helps me work with data in a table format. 
It helps me organize the real income and the predicted income so I can compare them.

---

4. Why do I use Pandas here?
5. 
= I use Pandas to **make a table with two columns** — one for the actual income and one for the model’s guess.  
This way, I can **see how close or different** the predictions are.

---

5. What is Matplotlib?
6. 
= Matplotlib is a tool for making charts in Python.  
It helps me **see my results as a graph**, instead of just looking at numbers.

---

6. Why do I use Matplotlib here?

= I use it to **draw a scatter plot**, which shows how close each prediction is to the real income.  
If the dots are close to the red line, it means the model guessed well.

---

7. What is Seaborn?

= Seaborn is a helper tool that makes **nicer-looking charts**, built on top of Matplotlib.  
It’s great for showing patterns in the data.

---

8. Why do I use Seaborn here?
9. 
= I use Seaborn to make **clear and simple charts** that show how well my model did.  

---

Code 

import pandas as pd  # I use pandas to open and work with my data like a table.

from sklearn.model_selection import train_test_split  # I use this to divide my data into training and testing parts.
from sklearn.ensemble import RandomForestRegressor  # I use this to build a smart model that predicts income.
from sklearn.metrics import mean_squared_error, r2_score  # I use these to check how good or bad the model’s guesses are.

# I open the CSV file that has all my cleaned and number-only freelancer data.
df = pd.read_csv("df_final_encoded.csv")

# I keep only the freelancers whose income is between $100 and $800.
# This helps me remove strange or extreme values that might confuse the model.
df = df[(df["monthly_income_usd"] >= 100) & (df["monthly_income_usd"] <= 800)]

# I create a new feature by multiplying hours worked and years of experience.
# This shows how much total skill/effort a person has built up.
df["skill_experience"] = df["hours_per_week"] * df["experience_years"]

# I separate the data into:
# X = all the things the model will look at to learn (like age, skills, hours, etc.)
# y = the one thing the model will try to guess (monthly income)
X = df.drop(columns=["monthly_income_usd", "freelancer_id", "secondary_skill", 
                     "internet_stability", "has_smartphone_only", "client_type_Local"])

y = df["monthly_income_usd"]  # I keep the monthly income as the target (the answer the model should guess).

# I split my data:
# 80% for training — this is what the model learns from.
# 20% for testing — this is like a quiz to see how good the model is at guessing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# I build my model using Random Forest (which is a group of decision trees working together).
# I tell it to use 200 trees and allow each tree to go 10 steps deep.
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# I train the model — this means it learns from the training data I gave it.
model.fit(X_train, y_train)

# I use the trained model to guess the incomes on the test data.
predictions = model.predict(X_test)

# I check how wrong the model’s guesses were by using:
# - Mean Squared Error (MSE): the smaller, the better.
# - R^2 Score: tells me how accurate the model is (closer to 100% is better).
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# I also calculate the average mistake in dollars — so I can see how far off the guesses were, on average.
avg_error = abs(y_test - predictions).mean()

# I print the results so I can understand how good my model is.
print("MSE:", round(mse, 2))  # I print the Mean Squared Error.
print("R^2 Score:", round(r2 * 100, 2), "%")  # I print the accuracy as a percentage.
print("Average Error:", round(avg_error, 2), "USD")  # I print the average prediction mistake in dollars.

Output 

Average Prediction Error:  77.37 USD
Biggest Prediction Mistakes: 
      Real Income  Predicted Income   Error
1922       657.28            312.94  344.34
4022       117.14            428.38  311.24
947        172.54            480.62  308.08
5373       395.04            678.37  283.33
4521       759.34            480.15  279.19
6108       158.33            433.72  275.39
435        712.15            438.03  274.12
420        758.49            487.53  270.96
6494       202.62            472.29  269.67
3909       386.95            650.73  263.78
Highest Incomes:
      Real Income  Predicted Income   Error
2328       791.74            569.60  222.14
6599       770.01            542.09  227.92
4521       759.34            480.15  279.19
420        758.49            487.53  270.96
1351       744.28            527.19  217.09
6470       741.81            590.64  151.17
5551       736.70            560.51  176.19
6471       730.53            586.42  144.11
2097       725.35            660.05   65.30
3770       717.73            539.89  177.84

The closer the dots are to the red line, the better the prediction.
If the dots are far above or below, the model guessed too high or too low. 
Mean Squared Error (MSE) :  9445.37
R^2 Score (% of accuracy):  53.02 %
Final Summary : MY CatBoost Model was about 53.02 % accurate. On average, it guessed about 77
 Model Used: RandomForestRegressor (200 trees, depth=10)

![image](https://github.com/user-attachments/assets/c17ee61d-97d4-493b-8cdf-4552282474ec)

Conclusion for Step 8: Test the Models (Evaluation)

The result I got was about **53.02% accuracy** using the **RandomForest model**.  
Earlier, I also tested **CatBoost**, and that model did slightly better — around **57.75% accurate**.

When I looked at the scatter plot, I noticed that **many points are close to the red line**, which means the model made good predictions. But there are also some dots **far above or below the line**, especially when income is very high or very low.

For example, one freelancer **actually earned $657**, but the model predicted only **$313** — that’s a mistake of over **$340**! So the model **understands the general trend**, but sometimes it still guesses **way too high or too low**.

The average prediction error was around **$77**, which means the model usually guesses within $77 of the correct income.  
That’s okay, but not perfect — it shows there’s still **room for improvement**, especially if I want to reach **70% or higher accuracy**.

Overall, I learned that **my model is on the right track**, but I need to **train it more** or use **even better features or data** to improve its predictions.

---

Step 9: **Asking and Answering Questions About the Project**  

**Project Title: Predicting Freelancer Income in Myanmar Using Skills and Platforms**

---

1. What challenges did I face, and how did I solve them?
   
One of the biggest challenges was that my model didn’t predict income very well at the beginning. The results were pretty random.  
I fixed that by **removing confusing columns**, **creating smarter features** (like platform + skill), and testing different models like **CatBoost** and **RandomForest**.  
It wasn’t perfect, but the predictions got a lot better after many tries.

---

2. How did the model perform compared to what I expected?
   
To be honest, I was expecting something like **70–90% accuracy**, but I got around **53%**.  
It was lower than I hoped, but I realized that **predicting income is tricky** — there are so many factors involved.  
Still, the model found useful patterns, like how **experience and hours worked** usually lead to higher income.

---

3. Which features were most important in predicting income?

The most helpful features were things like **years of experience**, **hours per week**, and which platform they used — like **Upwork** or **Fiverr**.  
I also created new ones, like **fiverr_graphic** and **skill_experience**, and those really helped the model understand things better.

---

4. What would I do next if I had more time?

If I had more time, I’d try stronger models like **LightGBM** or even **deep learning**.  
I’d also collect more detailed info — like **client reviews**, **project types**, or **ratings** — to help the model make smarter guesses.

---

5. How can this project help real freelancers?
 
It can give real advice to freelancers, like:  
 “Which skill should I focus on?”  
 “Which platform pays better?”  
 “Does experience really matter?”  
It could also help platforms or training programs figure out how to **support freelancers better**.

---

6. What did I learn from doing this project?

I learned that **clean data is everything**. Even small mistakes can ruin the model.  
I also learned how to test models, use visualizations, and explain data in a way that makes sense — not just to a computer, but to people too.

---

7. What was the hardest part of the project?
 
Definitely getting the model to be more accurate.  
Sometimes it would guess way too high or too low, and I had to keep tweaking features and trying again. It was frustrating — but I didn’t give up.

---

8. Were there any surprises in the results?

Yes! I thought working with **international clients** or having a **higher education level** would mean higher income — but the data showed that wasn't always true.  
Instead, things like **hours worked** and **experience** had a bigger impact.

---

9. If I could do it again, what would I do differently?
   
I’d spend more time cleaning the data, test more models, and maybe even make a simple tool where people could enter their info and get a prediction of their expected income.

---

10. How could this help in the future?
 
This project could help **job websites**, **schools**, or even **youth programs** in Myanmar.  
It could guide people toward **skills that pay more**, or show which cities need **better internet access** to succeed online.

---

11. Why did I choose this project?

I chose this because I know a lot of people — especially young people in Myanmar — want to work online, but don’t know where to start.  
I wanted to build something that’s not just for school, but something that could **actually help people** make better choices.  
If this project can help someone earn more online — even just one person — I’ll be proud.

---

Final Reflection  
In the end, this wasn’t just a coding project — it was about using real data to **help real people**.  
And that’s what made it meaningful for me.

---
