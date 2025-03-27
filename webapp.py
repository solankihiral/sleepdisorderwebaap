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

    X=data.iloc[:,:-1]
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
    
        
    if choice=="Login":
        Email = st.sidebar.text_input("Email")
        Password = st.sidebar.text_input("Password",type="password")
        b1=st.sidebar.checkbox("Login")
        if b1:
            regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if re.fullmatch(regex, Email):
                create_usertable()
                if Email=='a@a.com' and Password=='123':
                    st.success("Logged In as {}".format("Admin"))
                    Email=st.text_input("Delete Email")
                    if st.button('Delete'):
                        delete_user(Email)
                    user_result = view_all_users()
                    clean_db = pd.DataFrame(user_result,columns=["FirstName","LastName","Mobile","City","Email","password","Cpassword"])
                    st.dataframe(clean_db)
                else:
                    result = login_user(Email,Password)
                    if result:
                        st.success("Logged In as {}".format(Email))
                        menu2 = ["K-Nearest Neighbors", "SVM",
                                 "Decision Tree", "Random Forest",
                                 "Naive Bayes","ExtraTreesClassifier","VotingClassifier"]
                        choice2 = st.selectbox("Select ML",menu2)
                        
                        gd = ['Male', 'Female']
                        Gender=st.selectbox("Select Gender",gd)
                                          
                        Age=float(st.slider('age Value', 27, 60))
                        oc=['Software Engineer', 'Doctor', 'Sales Representative', 'Teacher',
                               'Nurse', 'Engineer', 'Accountant', 'Scientist', 'Lawyer',
                               'Salesperson', 'Manager']
                        Occupation=st.selectbox("Select Occupation",oc)
                        SD=float(st.slider('Sleep Duration', 5.8, 8.5))
                        QS=float(st.slider('Quality of Sleep', 4, 9))
                        PL=float(st.slider('Physical Activity Level', 30, 90))
                        SL=float(st.slider('Stress Level', 3, 8))
                        bm=['Overweight', 'Normal', 'Obese', 'Normal Weight']
                        bmc=st.selectbox("BMI Category",bm)
                        bp=['126/83', '125/80', '140/90', '120/80', '132/87', '130/86',
                               '117/76', '118/76', '128/85', '131/86', '128/84', '115/75',
                               '135/88', '129/84', '130/85', '115/78', '119/77', '121/79',
                               '125/82', '135/90', '122/80', '142/92', '140/95', '139/91',
                               '118/75']
                        bps=st.selectbox("Blood Pressure",bp)
                        HR=float(st.slider('Heart Rate', 65, 86))
                        DS=float(st.slider('Daily Steps', 3000, 10000))
    
                        
                        my_array=[Gender, Age, Occupation, SD, QS,
                               PL,SL,bmc,bps,HR,DS]
                        
                        b2=st.button("Predict")
                        model=pickle.load(open("model.pkl",'rb'))
                                               
                        if b2:                        
                            df = pd.DataFrame([my_array], 
                                              columns=['Gender', 'Age', 'Occupation', 'Sleep Duration', 'Quality of Sleep',
                                                     'Physical Activity Level', 'Stress Level', 'BMI Category',
                                                     'Blood Pressure', 'Heart Rate', 'Daily Steps'])
                            category_colums=['Gender','Occupation','BMI Category','Blood Pressure']
                            encoder=pickle.load(open("encoder.pkl",'rb'))
                            df[category_colums] = df[category_colums].apply(encoder.fit_transform)
                            tdata=df.to_numpy()
                            #st.write(tdata)
                            if choice2=="K-Nearest Neighbors":
                                test_prediction = model[0].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)
                            if choice2=="SVM":
                                test_prediction = model[1].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)                 
                            if choice2=="Decision Tree":
                                test_prediction = model[2].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)
                            if choice2=="Random Forest":
                                test_prediction = model[3].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)
                            if choice2=="Naive Bayes":
                                test_prediction = model[4].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)
                            if choice2=="ExtraTreesClassifier":
                                test_prediction = model[5].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)
                            if choice2=="VotingClassifier":
                                test_prediction = model[6].predict(tdata)
                                query=test_prediction[0]
                                st.success(query)
                                
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
                
            
    
            