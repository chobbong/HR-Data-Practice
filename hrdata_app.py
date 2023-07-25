import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

st.write("""
# Wage Analysis

This app predicts the **Wage**!
""")
st.write('---')

df = pd.read_csv('wage_example_1.csv')

# Define a dictionary for translating position names from Korean to English
# This is a simple mapping and may not reflect actual position names in all organizations
position_translation = {
    "사원": "Staff",
    "대리": "Assistant Manager",
    "과장": "Manager",
    "차장": "Deputy General Manager",
    "부장": "General Manager",
    "이사": "Director",
    "상무": "Executive Director",
    "전무": "Managing Director",
    "사장": "CEO",
}

# Replace the Korean position names with English names
df['Position'] = df['Position'].map(position_translation)

# Count the number of employees in each position
position_counts = df['Position'].value_counts()
team_counts = df['Team'].value_counts(normalize=True)
# Calculate the proportion of employees in each position
position_proportions = position_counts / position_counts.sum()
team_proportions = team_counts / team_counts.sum()




# Remove the commas from the 'Total-Payment-Amount' column and convert it to integers
df['Total-Payment-Amount'] = df['Total-Payment-Amount'].str.replace(',', '').astype(int)

# Convert 'DateOfEntry' to datetime format
df['DateOfEntry'] = pd.to_datetime(df['DateOfEntry'])

# Calculate the years of entry
df['YearsOfEntry'] = (pd.Timestamp.now() - df['DateOfEntry']).dt.days / 365

if st.button('Predict Total Pay Amount after 5 years **(Click here)**'):
    # Prepare the data for training
    X = df[['YearsOfEntry']]
    y = df['Total-Payment-Amount']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize a Linear Regression model
    lr = LinearRegression()

    # Train the model
    lr.fit(X_train, y_train)

    # Make predictions
    y_pred = lr.predict(X_test)

    # Calculate and display the R2 score
    r2 = r2_score(y_test, y_pred)
    # st.write(f'R2 Score: {r2}')

    # Predict the 'Total-Payment-Amount' for an increase of 5 years in 'YearsOfEntry'
    years_increase = 5
    new_years_of_entry = X_test.mean() + years_increase
    new_total_payment = lr.predict([new_years_of_entry.values])
    new_total_payment[0] =  int(new_total_payment[0]) 
    st.write(f'Predicted Total Pay Amount for an increase of {years_increase} years in Years of Entry:')
    st.write(f'{new_total_payment[0]:,} won')
    
# Define a function to categorize years of entry
def categorize_years(years):
    if years < 5:
        return 'Less than 5 years'
    elif 5 <= years < 10:
        return '5-10 years'
    elif 10 <= years < 15:
        return '10-15 years'
    elif 15 <= years < 20:
        return '15-20 years'
    else:
        return 'More than 20 years'
# Apply the function to the 'YearsOfEntry' column
df['EntryCategory'] = df['YearsOfEntry'].apply(categorize_years)

st.write('---')

graph_choice = st.selectbox('Select a graph', ['Position Distribution', 'Team Distribution'])

if graph_choice == 'Position Distribution':
    # Create a pie chart of position proportions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(position_proportions, labels=position_proportions.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Positions')
    st.pyplot(fig)

elif graph_choice == 'Team Distribution':
    # Create a pie chart of team proportions
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(team_proportions, labels=team_proportions.index, autopct='%1.1f%%')
    ax.set_title('Distribution of Teams')
    st.pyplot(fig)

st.write('---')

graph_choice = st.selectbox('Check the monthly pay according to your position or years of service ', ['Position vs Wage', 'Years of Entry vs Wage','Team vs Wage'])

if graph_choice == 'Position vs Wage':
    # Compute the average 'Total-Payment-Amount' for each position
    average_payment = df.groupby('Position')['Total-Payment-Amount'].mean()
    # Sort the average_payment Series in descending order
    average_payment_sorted = average_payment.sort_values(ascending=True)

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(average_payment_sorted.index, average_payment_sorted.values, color='steelblue')

    # Add the average payment values on top of each bar
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval, bar.get_y() + bar.get_height()/2, f"{int(xval):,} won"+" ", va='center', ha='right', color='white')

    plt.title('Average Total Pay Amount by Position')
    plt.xlabel('Average Total Pay Amount')
    plt.ylabel('Position')
    p1 = plt
    st.pyplot(p1)




elif graph_choice == 'Years of Entry vs Wage':

    # Compute the average 'Total-Payment-Amount' for each category
    average_payment_by_entry_category = df.groupby('EntryCategory')['Total-Payment-Amount'].mean()

    # Order the index by our specific order
    order = ['Less than 5 years', '5-10 years', '10-15 years', '15-20 years', 'More than 20 years']
    average_payment_by_entry_category = average_payment_by_entry_category.loc[order]
    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(average_payment_by_entry_category.index, average_payment_by_entry_category.values, color='steelblue')

    # Add the average payment values on top of each bar
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval, bar.get_y() + bar.get_height()/2, f"{int(xval):,} won"+" ", va='center', ha='right', color='white')

    plt.title('Average Total Pay Amount by Years of Entry')
    plt.xlabel('Average Total Pay Amount')
    plt.ylabel('Years of Entry')
    p2 = plt
    st.pyplot(p2)

elif graph_choice == 'Team vs Wage':
    # Compute the average 'Total-Payment-Amount' for each team
    average_payment_by_team = df.groupby('Team')['Total-Payment-Amount'].mean()

    # Sort the average_payment Series in descending order
    average_payment_by_team_sorted = average_payment_by_team.sort_values(ascending=True)

    # Create a horizontal bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(average_payment_by_team_sorted.index, average_payment_by_team_sorted.values, color='steelblue')

    # Add the average payment values on top of each bar
    for bar in bars:
        xval = bar.get_width()
        plt.text(xval, bar.get_y() + bar.get_height()/2, f"{int(xval):,} won"+" ", va='center', ha='right', color='white')

    plt.title('Average Total Pay  Amount by Team')
    plt.xlabel('Average Total Pay Amount')
    plt.ylabel('Team')
    st.pyplot(plt)

st.write('---')

st.write('Scatter plot between Years of Entry and Total Pay Amount')
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df['YearsOfEntry'], df['Total-Payment-Amount'], alpha=0.5)
ax.set_title('Scatter plot: Years of Entry vs Total Pay  Amount')
ax.set_xlabel('Years of Entry')
ax.set_ylabel('Total Payment Amount (won)')

st.pyplot(plt)

st.write('---')

st.write('distribution of Total Pay Amount for each team')
# Box plot between 'Team' and 'Total-Payment-Amount'
plt.figure(figsize=(10, 8))
ax = sns.boxplot(x='Team', y='Total-Payment-Amount', data=df)
plt.title('Box plot: Team vs Total Pay Amount')
plt.xlabel('Team')
plt.ylabel('Total Pay Amount (won)')
plt.xticks(rotation=90)

st.pyplot(plt)


# ...previous code...


    # # Plot the regression line
    # fig, ax = plt.subplots()
    # ax.scatter(X_test, y_test, color='blue')
    # ax.plot(X_test, y_pred, color='red')
    # ax.set_title('Years of Entry vs Total Payment Amount')
    # ax.set_xlabel('Years of Entry')
    # ax.set_ylabel('Total Payment Amount')
    # st.pyplot(fig)

# # 특성 데이터와 타겟 데이터를 분리합니다.
# features = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# target = raw_df.values[1::2, 2]

# # 특성 데이터를 DataFrame 형식으로 변환합니다.
# # 여기서는 데이터셋의 원래 특성 이름을 사용해야 하는데,
# # 이 정보가 제공되지 않았으므로 일단 'feature_i' 형식의 이름을 사용합니다.
# feature_names = [f'feature_{i}' for i in range(features.shape[1])]
# X = pd.DataFrame(features, columns=feature_names)

# # 타겟 데이터를 DataFrame 형식으로 변환합니다.
# # 원래 데이터셋의 타겟 변수 이름인 "MEDV"를 사용합니다.
# Y = pd.DataFrame(target, columns=["MEDV"])

# # Sidebar
# # Header of Specify Input Parameters
# st.sidebar.header('Specify Input Parameters')

# def user_input_features():
#     CRIM = st.sidebar.slider('CRIM', X.CRIM.min(), X.CRIM.max(), X.CRIM.mean())
#     ZN = st.sidebar.slider('ZN', X.ZN.min(), X.ZN.max(), X.ZN.mean())
#     INDUS = st.sidebar.slider('INDUS', X.INDUS.min(), X.INDUS.max(), X.INDUS.mean())
#     CHAS = st.sidebar.slider('CHAS', X.CHAS.min(), X.CHAS.max(), X.CHAS.mean())
#     NOX = st.sidebar.slider('NOX', X.NOX.min(), X.NOX.max(), X.NOX.mean())
#     RM = st.sidebar.slider('RM', X.RM.min(), X.RM.max(), X.RM.mean())
#     AGE = st.sidebar.slider('AGE', X.AGE.min(), X.AGE.max(), X.AGE.mean())
#     DIS = st.sidebar.slider('DIS', X.DIS.min(), X.DIS.max(), X.DIS.mean())
#     RAD = st.sidebar.slider('RAD', X.RAD.min(), X.RAD.max(), X.RAD.mean())
#     TAX = st.sidebar.slider('TAX', X.TAX.min(), X.TAX.max(), X.TAX.mean())
#     PTRATIO = st.sidebar.slider('PTRATIO', X.PTRATIO.min(), X.PTRATIO.max(), X.PTRATIO.mean())
#     B = st.sidebar.slider('B', X.B.min(), X.B.max(), X.B.mean())
#     LSTAT = st.sidebar.slider('LSTAT', X.LSTAT.min(), X.LSTAT.max(), X.LSTAT.mean())
#     data = {'CRIM': CRIM,
#             'ZN': ZN,
#             'INDUS': INDUS,
#             'CHAS': CHAS,
#             'NOX': NOX,
#             'RM': RM,
#             'AGE': AGE,
#             'DIS': DIS,
#             'RAD': RAD,
#             'TAX': TAX,
#             'PTRATIO': PTRATIO,
#             'B': B,
#             'LSTAT': LSTAT}
#     features = pd.DataFrame(data, index=[0])
#     return features

# df = user_input_features()

# # Main Panel

# # Print specified input parameters
# st.header('Specified Input parameters')
# st.write(df)
# st.write('---')

# # Build Regression Model
# model = RandomForestRegressor()
# model.fit(X, Y)
# # Apply Model to Make Prediction
# prediction = model.predict(df)

# st.header('Prediction of MEDV')
# st.write(prediction)
# st.write('---')

# # Explaining the model's predictions using SHAP values
# # https://github.com/slundberg/shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# st.header('Feature Importance')
# plt.title('Feature importance based on SHAP values')
# shap.summary_plot(shap_values, X)
# st.pyplot(bbox_inches='tight')
# st.write('---')

# plt.title('Feature importance based on SHAP values (Bar)')
# shap.summary_plot(shap_values, X, plot_type="bar")
# st.pyplot(bbox_inches='tight')
