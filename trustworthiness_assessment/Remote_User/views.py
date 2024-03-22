from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,detect_trust_type,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def index(request):
    return render(request, 'RUser/index.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city,address=address,gender=gender)

        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html',{'object':obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Product_Review_Trust(request):
    if request.method == "POST":

        if request.method == "POST":

            reviewerID= request.POST.get('reviewerID')
            Product_Id= request.POST.get('Product_Id')
            reviewerName= request.POST.get('reviewerName')
            helpful= request.POST.get('helpful')
            reviewText= request.POST.get('reviewText')
            overall_rating= request.POST.get('overall_rating')
            summary= request.POST.get('summary')
            unixReviewTime= request.POST.get('unixReviewTime')
            reviewTime= request.POST.get('reviewTime')
            day_diff= request.POST.get('day_diff')
            helpful_yes= request.POST.get('helpful_yes')
            total_vote= request.POST.get('total_vote')


        df = pd.read_csv('Product_Reviews.csv')

        def apply_response(total_vote):
            if (total_vote == 0):
                return 0  # No Trust
            elif (total_vote > 0 and total_vote < 50):
                return 1  # Average Trust
            elif (total_vote > 50):
                return 2  # Good Trust

        df['Results'] = df['total_vote'].apply(apply_response)

        cv = CountVectorizer()
        X = df['reviewText'].apply(str)
        y = df['Results']

        print("Review")
        print(X)
        print("Results")
        print(y)

        X = cv.fit_transform(X)


        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB

        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print("ACCURACY")
        print(naivebayes)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_nb))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print("ACCURACY")
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression

        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        print("Decision Tree Classifier")
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        dtcpredict = dtc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, dtcpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, dtcpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, dtcpredict))
        models.append(('DecisionTreeClassifier', dtc))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        reviewText1 = [reviewText]
        vector1 = cv.transform(reviewText1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if (prediction == 0):
            val = 'No Trust'
        elif (prediction == 1):
            val = 'Average Trust'
        elif (prediction == 2):
            val = 'Good Trust'

        print(val)
        print(pred1)

        detect_trust_type.objects.create(reviewerID=reviewerID,Product_Id=Product_Id,reviewerName=reviewerName,helpful=helpful,reviewText=reviewText,
        overall_rating=overall_rating,
        summary=summary,
        unixReviewTime=unixReviewTime,
        reviewTime=reviewTime,
        day_diff=day_diff,
        helpful_yes=helpful_yes,
        total_vote=total_vote,
        Prediction=val)

        return render(request, 'RUser/Predict_Product_Review_Trust.html',{'objs': val})
    return render(request, 'RUser/Predict_Product_Review_Trust.html')



