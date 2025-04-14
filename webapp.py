import streamlit as st
import re
import sqlite3 
import pickle
import pandas as pd
st.set_page_config(page_title="Sleep Disorder", page_icon="fevicon.png", layout="centered", initial_sidebar_state="auto", menu_items=None)
import os
file_path="model.pkl"
if not os.path.exists(file_path):
    data=pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    data=data.dropna(how='any')
    data=data.drop(["Person ID"],axis=1)

    #Label encoding
    category_colums=['Gender','Occupation','BMI Category','Blood Pressure']
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data[category_colums] = data[category_colums].apply(encoder.fit_transform)

    X=data[['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep','BMI Category','Blood Pressure', 'Heart Rate', 'Daily Steps']]
    y=data.iloc[:,-1]
    #array Conver
    X=X.to_numpy()

    #spilit data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    import warnings
    warnings.filterwarnings("ignore")
    names = ["K-Nearest Neighbors", "SVM",
             "Decision Tree", "Random Forest",
             "Naive Bayes","ExtraTreesClassifier","VotingClassifier"]

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    from sklearn.ensemble import VotingClassifier

    classifiers = [
        KNeighborsClassifier(),
        LinearSVC(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GaussianNB(),
        ExtraTreesClassifier(),
        VotingClassifier(estimators=[('DT', DecisionTreeClassifier()), ('rf', RandomForestClassifier()), ('et', ExtraTreesClassifier())], voting='hard')]

    clfF=[]
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        print(name)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        print('--------------------------------------------------------------')
        clfF.append(clf)
    pickle.dump(clfF, open("model.pkl", 'wb'))  
    pickle.dump(encoder, open("encoder.pkl",'wb'))    
else:

    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    # DB  Functions
    def create_usertable():
        c.execute('CREATE TABLE IF NOT EXISTS userstable(FirstName TEXT,LastName TEXT,Mobile TEXT,City TEXT,Email TEXT,password TEXT,Cpassword TEXT)')
    def add_userdata(FirstName,LastName,Mobile,City,Email,password,Cpassword):
        c.execute('INSERT INTO userstable(FirstName,LastName,Mobile,City,Email,password,Cpassword) VALUES (?,?,?,?,?,?,?)',(FirstName,LastName,Mobile,City,Email,password,Cpassword))
        conn.commit()
    def login_user(Email,password):
        c.execute('SELECT * FROM userstable WHERE Email =? AND password = ?',(Email,password))
        data = c.fetchall()
        return data
    def view_all_users():
    	c.execute('SELECT * FROM userstable')
    	data = c.fetchall()
    	return data
    def delete_user(Email):
        c.execute("DELETE FROM userstable WHERE Email="+"'"+Email+"'")
        conn.commit()
    
    
    menu = ["Home","Login","SignUp"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice=="Home":
        st.markdown(
            """
            <h2 style="color:black">Sleep Disorder Prediction System</h2>
            <h1>    </h1>
            <p align="justify">
            <b style="color:black">The Sleep Disorder Prediction System is an innovative application of artificial intelligence and machine learning algorithms designed to predict and diagnose various sleep disorders in individuals. By analyzing patterns and data related to sleep behavior, such as duration, quality, and disturbances, this system can accurately identify potential sleep disorders such as insomnia, sleep apnea, narcolepsy, and restless leg syndrome. Through the integration of wearable devices, smart sensors, and advanced data analytics, the Sleep Disorder Prediction System offers personalized insights and recommendations for improving sleep hygiene and overall well-being. This technology not only enhances early detection and treatment of sleep disorders but also empowers individuals to take proactive measures towards achieving better sleep health.</b>
            </p>
            """
            ,unsafe_allow_html=True)
    
    
    # Initialize session state variables if they don't exist
    def initialize_session_state():
        if "Gender" not in st.session_state:
            st.session_state["Gender"] = "Male"
        if 'Age' not in st.session_state:
            st.session_state['Age'] = 25  # Default age as an integer
        if 'Weight' not in st.session_state:
            st.session_state['Weight'] = 70.0
        if 'Height' not in st.session_state:
            st.session_state['Height'] = 170.0
        if 'Occupation' not in st.session_state:
            st.session_state['Occupation'] = 'Software Engineer'
        if 'Sleep Duration' not in st.session_state:
            st.session_state['Sleep Duration'] = 5.8
        if 'Sleep Quality' not in st.session_state:
            st.session_state['Sleep Quality'] = 'Very Poor'
        if 'Systolic' not in st.session_state:
            st.session_state['Systolic'] = 80
        if 'Diastolic' not in st.session_state:
            st.session_state['Diastolic'] = 40
        if 'Heart Rate' not in st.session_state:
            st.session_state['Heart Rate'] = 65.0
        if 'Daily Steps' not in st.session_state:
            st.session_state['Daily Steps'] = 3000.0
        if 'reset' not in st.session_state:
            st.session_state['reset'] = False
    
    # ------------------ RESET FUNCTION ------------------
    def reset_form():
        keys_to_reset = [
            "Gender", "Age", "Weight", "Height", "Occupation",
            "Sleep Duration", "Sleep Quality", "Systolic", "Diastolic",
            "Heart Rate", "Daily Steps"
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

    
    # ------------------ MAIN APP ------------------
    initialize_session_state()  # Initialize session state variables
    
    if choice == "Login":
        Email = st.sidebar.text_input("Email")
        Password = st.sidebar.text_input("Password", type="password")
        b1 = st.sidebar.checkbox("Login")
    
        if b1:
            regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if re.fullmatch(regex, Email):
                create_usertable()
                if Email == 'a@a.com' and Password == '123':
                    st.success(f"Logged In as Admin")
                    Email = st.text_input("Delete Email")
                    if st.button('Delete'):
                        delete_user(Email)
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result, columns=["FirstName", "LastName", "Mobile", "City", "Email", "password", "Cpassword"])
                    st.dataframe(clean_db)
                else:
                    result = login_user(Email, Password)
                    if result:
                        st.success(f"Logged In as {Email}")
    
                        # ------------------ FORM UI ------------------
    
                        model_choice = st.selectbox("Select ML", ["VotingClassifier"])
    
                        # Gender: Initialize the session state and use it with key parameter
                        if 'Gender' not in st.session_state:
                            st.session_state['Gender'] = 'Male'  # Default value
                        Gender = st.selectbox("Select Gender", ['Male', 'Female'], key="Gender", index=['Male', 'Female'].index(st.session_state["Gender"]))
    
                        # Age: Use number input for Age, enforce integer type
                        if 'Age' not in st.session_state:
                            st.session_state['Age'] = 25  # Default age
                        Age = st.number_input("Enter Age Value", min_value=1, value=st.session_state["Age"], key="Age", step=1)
    
                        # Weight: Use number input for Weight
                        if 'Weight' not in st.session_state:
                            st.session_state['Weight'] = 70  # Default weight
                        weight = st.number_input("Enter your weight (kg)", min_value=1.0, value=st.session_state["Weight"], key="Weight")
    
                        # Height: Use number input for Height
                        if 'Height' not in st.session_state:
                            st.session_state['Height'] = 170  # Default height
                        height = st.number_input("Enter your height (cm)", min_value=1.0, value=st.session_state["Height"], key="Height")
    
                        # Calculate BMI Category
                        if weight and height:
                            height_m = height / 100
                            bmi = weight / (height_m ** 2)
                            if bmi < 18.5:
                                bmc = "Underweight"
                            elif 18.5 <= bmi < 25:
                                bmc = "Normal weight"
                            elif 25 <= bmi < 30:
                                bmc = "Overweight"
                            else:
                                bmc = "Obese"
                        else:
                            bmc = "Unknown"
    
                        # Occupation
                        oc = [
                            'Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
                            'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer',
                            'Salesperson', 'Manager'
                        ]
                        Occupation = st.selectbox("Select Occupation", oc, key="Occupation")
    
                        # Sleep Duration
                        if 'Sleep Duration' not in st.session_state:
                            st.session_state["Sleep Duration"] = 7.0  # Default value
                        SD = float(st.slider("Enter Sleep Duration", 5.8, 8.5, value=st.session_state["Sleep Duration"], key="Sleep Duration"))
    
                        # Sleep Quality
                        sleep_quality_options = {
                            "Very Poor": 4,
                            "Poor": 5,
                            "Average": 6,
                            "Good": 7,
                            "Very Good": 8,
                            "Excellent": 9
                        }
                        selected_quality = st.selectbox("Select Quality of Sleep", list(sleep_quality_options.keys()), key="Sleep Quality")
                        QS = float(sleep_quality_options[selected_quality])
    
                        # Blood Pressure
                        col1, col2 = st.columns(2)
                        with col1:
                            systolic = st.number_input("Blood", 80, 200, value=st.session_state["Systolic"], key="Systolic")
                        with col2:
                            diastolic = st.number_input("Presure", 40, 130, value=st.session_state["Diastolic"], key="Diastolic")
                        bps = f"{int(systolic)}/{int(diastolic)}"
    
                        # Heart Rate and Steps
                        HR = float(st.number_input("Enter Heart Rate", min_value=1.0, value=st.session_state["Heart Rate"], key="Heart Rate"))
                        DS = float(st.number_input("Daily Steps", min_value=1.0, value=st.session_state["Daily Steps"], key="Daily Steps"))
    
                        # Final data array
                        my_array = [Gender, Age, Occupation, SD, QS, bmc, bps, HR, DS]
    
                        # Predict Button (after Reset button)
                        if st.button("Predict"):
                            model = pickle.load(open("model.pkl", 'rb'))
                            df = pd.DataFrame([my_array], columns=[
                                'Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                                'BMI Category', 'Blood Pressure', 'Heart Rate', 'Daily Steps'
                            ])
                            # Encode categorical data
                            category_columns = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure']
                            encoder = pickle.load(open("encoder.pkl", 'rb'))
                            df[category_columns] = df[category_columns].apply(encoder.fit_transform)
                            tdata = df.to_numpy()
    
                            if model_choice == "VotingClassifier":
                                prediction = model[6].predict(tdata)
                                if prediction[0]=="Sleep Apnea":
                                    prd="Sleep Apnea: A serious sleep disorder in which breathing repeatedly stops and starts during sleep."
                                else:
                                    prd="Insomnia: Difficulty falling asleep, staying asleep, or waking up too early and not being able to get back."
                                st.success(f"The system has detected signs of : {prd}")
    
                        # Reset Button (after Predict button)
                        if st.button("Reset All Selections"):
                            reset_form()  # Resets the form to default values
                    else:
                        st.warning("Incorrect Email/Password")
            else:
                st.warning("Not Valid Email")




                    
               
    if choice=="SignUp":
        Fname = st.text_input("First Name")
        Lname = st.text_input("Last Name")
        Mname = st.text_input("Mobile Number")
        Email = st.text_input("Email")
        City = st.text_input("City")
        Password = st.text_input("Password",type="password")
        CPassword = st.text_input("Confirm Password",type="password")
        b2=st.button("SignUp")
        if b2:
            pattern=re.compile("(0|91)?[7-9][0-9]{9}")
            regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if Password==CPassword:
                if (pattern.match(Mname)):
                    if re.fullmatch(regex, Email):
                        create_usertable()
                        add_userdata(Fname,Lname,Mname,City,Email,Password,CPassword)
                        st.success("SignUp Success")
                        st.info("Go to Logic Section for Login")
                    else:
                        st.warning("Not Valid Email")         
                else:
                    st.warning("Not Valid Mobile Number")
            else:
                st.warning("Pass Does Not Match")
                
            
    
            
