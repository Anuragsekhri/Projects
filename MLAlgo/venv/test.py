import tkinter as Tk
import pandas
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import time

################################################################
class OtherLinear(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("Decision Tree")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=25)


        Heading = Tk.Label(self, text="Linear Regression Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=2, columnspan=3)

        # Detail = Tk.Label(self,text="Prediction of Miles/Gallon Of Car",bg="black",
        #                    height=4, fg="red", font=("Helvetica", 20, "bold"))
        # Detail.grid(row=2,column=2)

        url = "regre.csv"
        names = ['mpg', 'cylinder', 'displ', 'hpower', 'weight', 'acc', 'model', 'origin', 'name']

        dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
        array = dataframe.values
        X = array[:, 1:7]
        Y = array[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        y_pred = regressor.predict(X_test)
        intercept = (regressor.intercept_)
        coeff = regressor.coef_  # for every feature coeff

        y_pred = y_pred.astype(int)

        ex_var_score = explained_variance_score(y_test, y_pred)
        m_absolute_error = mean_absolute_error(y_test, y_pred)
        m_squared_error = mean_squared_error(y_test, y_pred)
        r_2_score = r2_score(y_test, y_pred)

        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111, projection='3d')

        a.scatter(dataframe['mpg'], dataframe['displ'], dataframe['weight'], c='blue', marker='o', alpha=0.5)
        a.set_xlabel('Mile/Gallon')
        a.set_ylabel('Displacement')
        a.set_zlabel('Weight')

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=4)

        f1 = Figure(figsize=(5,5), dpi=100)
        a1 = f1.add_subplot(111, projection='3d')

        a1.scatter(dataframe['mpg'], dataframe['cylinder'], dataframe['hpower'], c='blue', marker='o', alpha=0.5)
        a1.set_xlabel('Mile/Gallon')
        a1.set_ylabel('No of Cylinder')
        a1.set_zlabel('Hpower')

        canvas1 = FigureCanvasTkAgg(f, self)
        canvas1.draw()
        canvas1.get_tk_widget().grid(row=3, column=7)

        f2 = Figure(figsize=(5,5 ), dpi=100)
        a2 = f2.add_subplot(111, projection='3d')

        a2.scatter(dataframe['mpg'], dataframe['acc'], dataframe['model'], c='blue', marker='o', alpha=0.5)
        a2.set_xlabel('Mile/Gallon')
        a2.set_ylabel('Accerlation')
        a2.set_zlabel('Model')

        canvas2 = FigureCanvasTkAgg(f, self)
        canvas2.draw()
        canvas2.get_tk_widget().grid(row=3, column=11)



        Mean = Tk.Label(self, text="Explained Variance Score:   "+str(ex_var_score), anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=4)

        Mean1 = Tk.Label(self, text="Mean Absolute Error:   "+str(m_absolute_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean1.grid(row=6, column=5, columnspan=4 )

        Mean2 = Tk.Label(self, text="Mean Squared Error:    "+str(m_squared_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean2.grid(row=7, column=5, columnspan=4 )

        Mean3 = Tk.Label(self, text="R Squared Error:   "+str(r_2_score), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean3.grid(row=8, column=5, columnspan=4 )

        Mean4 = Tk.Label(self, text="For Prediction Of Miles Per Gallon Of Car", justify="left",
                         bg="black", fg="White",
                         font=("Helvetica", 17, "bold"))
        Mean4.grid(row=9, column=3, columnspan=4)



    # ----------------------------------------------------------------------
    def onClose(self):
        """"""

        self.destroy()

        # self.original_frame.show()

#############################################################


class OtherPoly2(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("Decision Tree")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=25)


        Heading = Tk.Label(self, text="Polynomial Regression ( Degree = 2 )  Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=2, columnspan=3)

        # Detail = Tk.Label(self,text="Prediction of Miles/Gallon Of Car",bg="black",
        #                    height=4, fg="red", font=("Helvetica", 20, "bold"))
        # Detail.grid(row=2,column=2)

        url = "regre.csv"
        names = ['mpg', 'cylinder', 'displ', 'hpower', 'weight', 'acc', 'model', 'origin', 'name']

        dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
        array = dataframe.values
        X = array[:, 1:7]
        Y = array[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        #regressor = LinearRegression()
        # regressor.fit(X_train, y_train)
        #
        # y_pred = regressor.predict(X_test)
        # intercept = (regressor.intercept_)
        # coeff = regressor.coef_  # for every feature coeff
        #
        # y_pred = y_pred.astype(int)

        poly = PolynomialFeatures(degree=2)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(X)

        clf = linear_model.LinearRegression()
        clf.fit(X_,Y)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(clf.predict(predict_))
        y_pred= clf.predict((predict_))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        ex_var_score = explained_variance_score(Y, y_pred)
        m_absolute_error = mean_absolute_error(Y, y_pred)
        m_squared_error = mean_squared_error(Y, y_pred)
        r_2_score = r2_score(Y, y_pred)

        f = Figure(figsize=(10,6), dpi=100)
        xx = list(range(0,len(X)))
        #a = f.add_subplot(111, projection='3d')
        a= f.add_subplot(111)
        #a.plot(xx,y_test,'ro',marker=".",label="True Value")
        #a.plot(xx,y_pred,'bo',marker=".",label="Predicted Value") #point plot
        a.plot(xx, Y, 'r', marker=".", label="True Value")
        a.plot(xx, y_pred, 'b', marker=".", label="Predicted Value")
        a.set_xlabel("Index")
        a.legend()
        a.set_ylabel("Y Value")
        #a.scatter(dataframe['mpg'], dataframe['displ'], dataframe['weight'], c='blue', marker='o', alpha=0.5)
        # a.set_xlabel('Mile/Gallon')
        # a.set_ylabel('Displacement')
        # a.set_zlabel('Weight')

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=4,rowspan=10)




        Mean = Tk.Label(self, text="Explained Variance Score:   "+str(ex_var_score), anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=2, column=8, columnspan=4)

        Mean1 = Tk.Label(self, text="Mean Absolute Error:   "+str(m_absolute_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean1.grid(row=3, column=8, columnspan=4 )

        Mean2 = Tk.Label(self, text="Mean Squared Error:    "+str(m_squared_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean2.grid(row=3, column=8, columnspan=4 )

        Mean3 = Tk.Label(self, text="R Squared Error:   "+str(r_2_score), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean3.grid(row=4, column=8, columnspan=4 )

        Mean4 = Tk.Label(self, text="For Prediction Of Miles Per Gallon Of Car", justify="left",
                         bg="black", fg="White",
                         font=("Helvetica", 17, "bold"))
        Mean4.grid(row=6, column=9, columnspan=4)



    # ----------------------------------------------------------------------
    def onClose(self):
        """"""

        self.destroy()

###########################################

class OtherPoly3(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("Decision Tree")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=25)


        Heading = Tk.Label(self, text="Polynomial Regression ( Degree = 3 )  Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=2, columnspan=3)

        # Detail = Tk.Label(self,text="Prediction of Miles/Gallon Of Car",bg="black",
        #                    height=4, fg="red", font=("Helvetica", 20, "bold"))
        # Detail.grid(row=2,column=2)

        url = "regre.csv"
        names = ['mpg', 'cylinder', 'displ', 'hpower', 'weight', 'acc', 'model', 'origin', 'name']

        dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
        array = dataframe.values
        X = array[:, 1:7]
        Y = array[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        #regressor = LinearRegression()
        # regressor.fit(X_train, y_train)
        #
        # y_pred = regressor.predict(X_test)
        # intercept = (regressor.intercept_)
        # coeff = regressor.coef_  # for every feature coeff
        #
        # y_pred = y_pred.astype(int)

        poly = PolynomialFeatures(degree=3)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(X)

        clf = linear_model.LinearRegression()
        clf.fit(X_,Y)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(clf.predict(predict_))
        y_pred= clf.predict((predict_))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        ex_var_score = explained_variance_score(Y, y_pred)
        m_absolute_error = mean_absolute_error(Y, y_pred)
        m_squared_error = mean_squared_error(Y, y_pred)
        r_2_score = r2_score(Y, y_pred)

        f = Figure(figsize=(10,6), dpi=100)
        xx = list(range(0,len(X)))
        #a = f.add_subplot(111, projection='3d')
        a= f.add_subplot(111)
        #a.plot(xx,y_test,'ro',marker=".",label="True Value")
        #a.plot(xx,y_pred,'bo',marker=".",label="Predicted Value") #point plot
        a.plot(xx, Y, 'r', marker=".", label="True Value")
        a.plot(xx, y_pred, 'b', marker=".", label="Predicted Value")
        a.set_xlabel("Index")
        a.legend()
        a.set_ylabel("Y Value")
        #a.scatter(dataframe['mpg'], dataframe['displ'], dataframe['weight'], c='blue', marker='o', alpha=0.5)
        # a.set_xlabel('Mile/Gallon')
        # a.set_ylabel('Displacement')
        # a.set_zlabel('Weight')

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=4,rowspan=10)




        Mean = Tk.Label(self, text="Explained Variance Score:   "+str(ex_var_score), anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=2, column=8, columnspan=4)

        Mean1 = Tk.Label(self, text="Mean Absolute Error:   "+str(m_absolute_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean1.grid(row=3, column=8, columnspan=4 )

        Mean2 = Tk.Label(self, text="Mean Squared Error:    "+str(m_squared_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean2.grid(row=3, column=8, columnspan=4 )

        Mean3 = Tk.Label(self, text="R Squared Error:   "+str(r_2_score), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean3.grid(row=4, column=8, columnspan=4 )

        Mean4 = Tk.Label(self, text="For Prediction Of Miles Per Gallon Of Car", justify="left",
                         bg="black", fg="White",
                         font=("Helvetica", 17, "bold"))
        Mean4.grid(row=6, column=9, columnspan=4)



    # ----------------------------------------------------------------------
    def onClose(self):
        """"""

        self.destroy()

################################################################

class OtherPoly4(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("Decision Tree")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=25)


        Heading = Tk.Label(self, text="Polynomial Regression ( Degree = 4 )  Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=2, columnspan=3)

        # Detail = Tk.Label(self,text="Prediction of Miles/Gallon Of Car",bg="black",
        #                    height=4, fg="red", font=("Helvetica", 20, "bold"))
        # Detail.grid(row=2,column=2)

        url = "regre.csv"
        names = ['mpg', 'cylinder', 'displ', 'hpower', 'weight', 'acc', 'model', 'origin', 'name']

        dataframe = pandas.read_csv(url, delim_whitespace=True, names=names)
        array = dataframe.values
        X = array[:, 1:7]
        Y = array[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

        #regressor = LinearRegression()
        # regressor.fit(X_train, y_train)
        #
        # y_pred = regressor.predict(X_test)
        # intercept = (regressor.intercept_)
        # coeff = regressor.coef_  # for every feature coeff
        #
        # y_pred = y_pred.astype(int)

        poly = PolynomialFeatures(degree=4)
        X_ = poly.fit_transform(X)
        predict_ = poly.fit_transform(X)

        clf = linear_model.LinearRegression()
        clf.fit(X_,Y)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(clf.predict(predict_))
        y_pred= clf.predict((predict_))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        ex_var_score = explained_variance_score(Y, y_pred)
        m_absolute_error = mean_absolute_error(Y, y_pred)
        m_squared_error = mean_squared_error(Y, y_pred)
        r_2_score = r2_score(Y, y_pred)

        f = Figure(figsize=(10,6), dpi=100)
        xx = list(range(0,len(X)))
        #a = f.add_subplot(111, projection='3d')
        a= f.add_subplot(111)
        #a.plot(xx,y_test,'ro',marker=".",label="True Value")
        #a.plot(xx,y_pred,'bo',marker=".",label="Predicted Value") #point plot
        a.plot(xx, Y, 'r', marker=".", label="True Value")
        a.plot(xx, y_pred, 'b', marker=".", label="Predicted Value")
        a.set_xlabel("Index")
        a.legend()
        a.set_ylabel("Y Value")
        #a.scatter(dataframe['mpg'], dataframe['displ'], dataframe['weight'], c='blue', marker='o', alpha=0.5)
        # a.set_xlabel('Mile/Gallon')
        # a.set_ylabel('Displacement')
        # a.set_zlabel('Weight')

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=2, column=4,rowspan=10)




        Mean = Tk.Label(self, text="Explained Variance Score:   "+str(ex_var_score), anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=2, column=8, columnspan=4)

        Mean1 = Tk.Label(self, text="Mean Absolute Error:   "+str(m_absolute_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean1.grid(row=3, column=8, columnspan=4 )

        Mean2 = Tk.Label(self, text="Mean Squared Error:    "+str(m_squared_error), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean2.grid(row=3, column=8, columnspan=4 )

        Mean3 = Tk.Label(self, text="R Squared Error:   "+str(r_2_score), anchor="e", justify="left",
                        bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean3.grid(row=4, column=8, columnspan=4 )

        Mean4 = Tk.Label(self, text="For Prediction Of Miles Per Gallon Of Car", justify="left",
                         bg="black", fg="White",
                         font=("Helvetica", 17, "bold"))
        Mean4.grid(row=6, column=9, columnspan=4)



    # ----------------------------------------------------------------------
    def onClose(self):
        """"""

        self.destroy()

########################################################################
class OtherFrameDecision(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen",True)
        self.title("Decision Tree")
        btn = Tk.Button(self, command=self.onClose,text="X" , width=5,height=2,bg="white",fg="red",font=("Helvetica",13,"bold"))
        btn.grid(row=0,column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:,0:5]
        Y = array[:,6]

        seed=7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(DecisionTreeClassifier(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(DecisionTreeClassifier(), X, Y)
        print("Decision Tree \n",cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3,column=12,columnspan=10)

        text="""
Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
or examples by starting at the root of the tree and moving through it until a leaf node.

    1) Decision node: specifies a test on a single attribute
    2) Leaf node: indicates the value of the target attribute 
    3) Arc/edge: split of one attribute
    4) Path: a disjunction of test to make the final decision 
    5) Entropy : A measure of homogeneity of the set of examples.
                 Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                 the entropy of set S relative to this binary classification is
                    
                        E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 
    
    6) Information gain measures the expected reduction in entropy, or uncertainty.
    
        """

        Heading = Tk.Label(self, text="Decision Tree Classifier Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9,columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5,columnspan=5,rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("Decision Tree Classifier", cv_results.mean(), cv_results.std())
        print(msg)

        text1="""
        
        1) Mean Scoring : %f
        2) Standard Deviation Scoring :  %f 
        
        
        """% (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                          font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)




    # ----------------------------------------------------------------------
    def onClose(self):
        """"""

        self.destroy()

        #self.original_frame.show()

########################################################################

########################################################################
class OtherFrameGNB(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("GaussianNB")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(GaussianNB(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(GaussianNB(), X, Y)
        print("Decision Tree \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="Gaussian Navie Bayes Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("GaussianNB", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
        #self.original_frame.show()


########################################################################
class OtherFrameBNB(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("BernoulliNB")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(BernoulliNB(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(BernoulliNB(), X, Y)
        print("BernoulliNB \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="Bernoulli Navie Bayes Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("BernoulliNB", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
#        self.original_frame.show()


########################################################################
########################################################################
class OtherFrameNB(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("MultinomialNB")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(MultinomialNB(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(MultinomialNB(), X, Y)
        print("MultinomialNB \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="Multinomial Navie Bayes Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("Multinomial Navie Bayes", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
 #       self.original_frame.show()


########################################################################
########################################################################
class OtherFrameLogistic(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("LogisticRegression")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(LogisticRegression(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(LogisticRegression(), X, Y)
        print("Decision Tree \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="Logistic Regression Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("Logistic Regression ", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
  #      self.original_frame.show()


########################################################################
########################################################################
class OtherFrameLDA(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("LinearDiscriminantAnalysis")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(LinearDiscriminantAnalysis(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(DecisionTreeClassifier(), X, Y)
        print("LinearDiscriminantAnalysis \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="Linear Discriminant AnalysisAlgorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("Linear Discriminant Analysis", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
   #     self.original_frame.show()


########################################################################
########################################################################
class OtherFrameKNC(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("KNeighborsClassifier")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(KNeighborsClassifier(), X, Y)
        print("Decision Tree \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="KNeighbors Classifier Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("KNeighborsClassifier", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
    #    self.original_frame.show()


########################################################################

########################################################################
class OtherFrameSVC(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("SVC")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btn.grid(row=0, column=30)

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)
        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]

        seed = 7
        scoring = 'accuracy'
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(SVC(), X, Y, cv=kfold, scoring=scoring)
        YPre = model_selection.cross_val_predict(SVC(), X, Y)
        print("SVC \n", cv_results);

        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.plot(cv_results);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=12, columnspan=10)

        text = """
        Decision tree is a classifier in the form of a tree structure.Decision trees classify instances
        or examples by starting at the root of the tree and moving through it until a leaf node.

            1) Decision node: specifies a test on a single attribute
            2) Leaf node: indicates the value of the target attribute 
            3) Arc/edge: split of one attribute
            4) Path: a disjunction of test to make the final decision 
            5) Entropy : A measure of homogeneity of the set of examples.
                         Given a set S of positive and negative examples of some target concept (a 2-class problem), 
                         the entropy of set S relative to this binary classification is

                                E(S) = - p(P)log2 p(P) – p(N)log2 p(N) 

            6) Information gain measures the expected reduction in entropy, or uncertainty.

                """

        Heading = Tk.Label(self, text="Support Vector Machine Algorithm : ", bg="black",
                           height=4, fg="red", font=("Helvetica", 20, "bold", "underline"))
        Heading.grid(row=1, column=9, columnspan=3)

        Detail = Tk.Label(self, text=text, anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        Detail.grid(row=1, column=5, columnspan=5, rowspan=3)

        # toolbar = NavigationToolbar2TkAgg(canvas, self)
        # toolbar.update()
        # canvas._tkcanvas.grid(row=5,column=6)

        msg = "%s: %f (%f)" % ("SVC", cv_results.mean(), cv_results.std())
        print(msg)

        text1 = """

                1) Mean Scoring : %f
                2) Standard Deviation Scoring :  %f 


                """ % (cv_results.mean(), cv_results.std())

        Mean = Tk.Label(self, text=text1, anchor="e", justify="left", bg="black", fg="yellow",
                        font=("Helvetica", 13, "bold"))
        Mean.grid(row=5, column=5, columnspan=5, rowspan=5)

    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
     #   self.original_frame.show()


########################################################################

class OtherAllClassi(Tk.Toplevel):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, original):
        """Constructor"""
        self.original_frame = original
        Tk.Toplevel.__init__(self)
        self.geometry("1250x600")
        self.configure(bg="black")
        self.attributes("-fullscreen", True)
        self.title("SVC")
        btn = Tk.Button(self, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        #btn.grid(row=0, column=0)

        btn.pack()

        url = "classi.csv"
        names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'Ans']
        dataframe = pandas.read_csv(url, names=names)

        array = dataframe.values
        X = array[:, 0:5]
        Y = array[:, 6]
        # prepare configuration for cross validation test harness
        seed = 7
        # prepare models
        models = []
        models.append(('SVM', SVC()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('GNB', GaussianNB()))
        models.append(('BNB', BernoulliNB()))
        models.append(('MNB', MultinomialNB()))
        models.append(('LR', LogisticRegression()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('DT', DecisionTreeClassifier()))


        # evaluate each model in turn
        results = []
        names = []
        #msgs=['LogisticRegression','LinearDiscriminantAnalysis','KNeighborsClassifier','DecisionTreeClassifier','GaussianNB','BernoulliNB','MultinomialNB','SVC']
        msgs=[]
        preds = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
            YPre = model_selection.cross_val_predict(model, X, Y)
            print(cv_results);
            preds.append(YPre);
            #plt.plot(cv_results[:-1]);
            #plt.show()
            results.append(cv_results)
            names.append(name)
            msg = "%s:  %f         %f" % (name, cv_results.mean(), cv_results.std())
            msgs.append(msg)
            print(msg)

        # 5555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555
        Detail1 = Tk.Label(self, text="", anchor="e", justify="left", bg="black", fg="blue",
                           font=("Helvetica", 17, "bold"))
        Detail1.pack()
        f = Figure(figsize=(5, 5), dpi=100)
        a = f.add_subplot(111)
        a.boxplot(results);
        a.set_xticklabels(names);
        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack()

        Detail3 = Tk.Label(self, text="                         Mean               Std Deviation", anchor="e", justify="left", bg="black", fg="blue",
                           font=("Helvetica", 14, "bold"))
        Detail3.pack()

        for x in msgs:
            Detail2 = Tk.Label(self, text=x, anchor="e", justify="left", bg="black", fg="yellow",
                           font=("Helvetica", 13, "bold"))
            Detail2.pack()



    # ----------------------------------------------------------------------
    def onClose(self):
        """"""
        self.destroy()
     #   self.original_frame.show()


########################################################################

class MyApp(object):
    """"""

    # ----------------------------------------------------------------------
    def __init__(self, parent):
        """Constructor"""
        self.root = parent
        self.root.title("Main frame")
        self.frame = Tk.Frame(parent,bg="black")
        self.frame.pack()


        text="""
The key to a fair comparison of machine learning algorithms is ensuring that each algorithm is evaluated in the same way on the same data.
In the example below 8 different algorithms are compared:

1. Decision Tree
2. Logistic Regression
3. Linear Discriminant Analysis 
4. K-Nearest Neighbors Classifier
5. Guassian Naive Bayes
6. Bernoulli Naive Bayes
7. Naive Bayes
8. Support Vector Machines
                
The 10-fold cross validation procedure is used to evaluate each algorithm, importantly configured with the same random seed to ensure that the same splits to the training data are 
performed and that each algorithms is evaluated in precisely the same way.

        """


        Heading = Tk.Label(self.frame,text="Machine Learning  Algorithms ",bg="black",height=4,fg="red",font=("Helvetica",20,"bold","underline"))
        Heading.grid(row=1,column=2,columnspan=4)
        btnClose = Tk.Button(self.frame, command=self.onClose, text="X", width=5, height=2, bg="white", fg="red",
                        font=("Helvetica", 13, "bold"))
        btnClose.grid(row=0, column=7)

        Detail1 = Tk.Label(self.frame, text="Classification", anchor="e", justify="left", bg="black", fg="blue",
                          font=("Helvetica", 17, "bold"))
        Detail1.grid(row=2, column=0, columnspan=10)

        Detail = Tk.Label(self.frame,text=text,anchor="e",justify="left",bg="black",fg="white",font=("Helvetica",13,"bold"))
        Detail.grid(row=3,column=0,columnspan=10)



        btn = Tk.Button(self.frame, text="Decision Tree", command=self.openFrameDecision , width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn.grid(row=4, column=0)

        btn1 = Tk.Button(self.frame, text="Logistic Regression", command=self.openFrameLogistic, width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn1.grid(row=4, column=1)

        btn2 = Tk.Button(self.frame, text="Linear Discriminant Analysis", command=self.openFrameLDA, width=25,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn2.grid(row=4, column=2)

        btn3 = Tk.Button(self.frame, text="KNeighbors Classifier", command=self.openFrameKNC, width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn3.grid(row=4, column=3)

        btn4 = Tk.Button(self.frame, text="Gaussian NB", command=self.openFrameGNB, width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn4.grid(row=4, column=4)

        btn5 = Tk.Button(self.frame, text="Bernoulli NB", command=self.openFrameBNB, width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn5.grid(row=4, column=5)

        btn6 = Tk.Button(self.frame, text="MultiNomial NB", command=self.openFrameNB, width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn6.grid(row=4, column=6)

        btn7 = Tk.Button(self.frame, text="SVC", command=self.openFrameSVC, width=22,bg="white",height=2,fg="blue",font=("Helvetica",10,"bold"))
        btn7.grid(row=4, column=7)

        xyz = Tk.Label(self.frame, text="", anchor="e", justify="left", bg="black", fg="white",
                          font=("Helvetica", 13, "bold"))
        xyz.grid(row=7, column=0, columnspan=10)

        btn8 = Tk.Button(self.frame, text="COMPARE ALL CLASSIFICATION ALGO", command=self.openFrameAll, width=32,bg="yellow",height=2,fg="black",font=("Helvetica",10,"bold"))
        btn8.grid(row=8, column=2,columnspan=4)

        xyz1 = Tk.Label(self.frame, text="", anchor="e", justify="left", bg="black", fg="white",
                       font=("Helvetica", 13, "bold"))
        xyz1.grid(row=9, column=0, columnspan=10)

        Detail2 = Tk.Label(self.frame, text="Regression", anchor="e", justify="left", bg="black", fg="blue",
                           font=("Helvetica", 17, "bold"))
        Detail2.grid(row=10, column=0, columnspan=10)

        text2 = """Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). 
                This technique is used for forecasting,time series modelling and finding the causal effect relationship between the variables."""


        xyz2 = Tk.Label(self.frame, text=text2, anchor="e", justify="left", bg="black", fg="white",
                        font=("Helvetica", 13, "bold"))
        xyz2.grid(row=11, column=0, columnspan=10)

        xyz3 = Tk.Label(self.frame, text="", anchor="e", justify="left", bg="black", fg="white",
                        font=("Helvetica", 13, "bold"))
        xyz3.grid(row=12, column=0, columnspan=10)


        btn9 = Tk.Button(self.frame, text="Linear Regression", command=self.openLinear, width=25, bg="white",
                        height=2, fg="blue", font=("Helvetica", 10, "bold"))
        btn9.grid(row=13, column=1)

        btn10 = Tk.Button(self.frame, text="Polynomial Reg. - Degree 2", command=self.openPoly2, width=25, bg="white",
                         height=2, fg="blue", font=("Helvetica", 10, "bold"))
        btn10.grid(row=13, column=2)

        btn11 = Tk.Button(self.frame, text="Polynomial Reg. - Degree 3", command=self.openPoly3, width=25,
                         bg="white", height=2, fg="blue", font=("Helvetica", 10, "bold"))
        btn11.grid(row=13, column=3)

        btn12 = Tk.Button(self.frame, text="Polynomial Reg. - Degree 4", command=self.openPoly4, width=25,
                         bg="white", height=2, fg="blue", font=("Helvetica", 10, "bold"))
        btn12.grid(row=13, column=4)

        btn13 = Tk.Button(self.frame, text="About DataSets Used", command=self.openFrameLDA, width=25,
                          bg="pink", height=2, fg="black", font=("Helvetica", 10, "bold"))
        btn13.grid(row=13, column=6)
    # ----------------------------------------------------------------------
    def hide(self):
        """"""
        self.root.withdraw()

    # ----------------------------------------------------------------------
    def openFrameDecision(self):
        """"""
        #self.hide()
        subFrame = OtherFrameDecision(self)

    # ----------------------------------------------------------------------
    def openFrameLDA(self):
        """"""
        #self.hide()
        subFrame = OtherFrameLDA(self)

    # ----------------------------------------------------------------------
    def openFrameKNC(self):
        """"""
        #self.hide()
        subFrame = OtherFrameKNC(self)

    # ----------------------------------------------------------------------
    def openFrameLogistic(self):
        """"""
        #self.hide()
        subFrame = OtherFrameLogistic(self)

    # ----------------------------------------------------------------------
    def openFrameBNB(self):
        """"""
        #self.hide()
        subFrame = OtherFrameBNB(self)

    # ----------------------------------------------------------------------
    def openFrameGNB(self):
        """"""
        #self.hide()
        subFrame = OtherFrameGNB(self)

    # ----------------------------------------------------------------------
    def openFrameNB(self):
        """"""
        #self.hide()
        subFrame = OtherFrameNB(self)

    # ----------------------------------------------------------------------
    def openFrameSVC(self):
        """"""
        #self.hide()
        subFrame = OtherFrameSVC(self)

    # ----------------------------------------------------------------------
    def openFrameAll(self):
        """"""
        #self.hide()
        subFrame = OtherAllClassi(self)

    def openLinear(self):
        subFrame = OtherLinear(self)

    def openPoly2(self):
        subFrame = OtherPoly2(self)
    def openPoly3(self):
        subFrame = OtherPoly3(self)
    def openPoly4(self):
        subFrame = OtherPoly4(self)


    def show(self):
        """"""
        self.root.update()
        self.root.deiconify()

    def onClose(self):
        """"""
        self.frame.destroy()
        self.hide()



# ----------------------------------------------------------------------
if __name__ == "__main__":
    root = Tk.Tk()
    root.attributes("-fullscreen",True)
    root.configure(bg="black")
    root.geometry("1400x600")
    app = MyApp(root)
    root.mainloop()

