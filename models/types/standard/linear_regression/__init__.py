"""
Contains a hybrid/residual linear regression and decision tree model
"""
# flake8: noqa: C901

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
import numpy as np
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LinearRegression():
    def __init__(self, degree=1, data=None):
        self.degree = degree
        self.scaler = StandardScaler()
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=True)
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        self.linreg = LinearRegression()
        self.tree = DecisionTreeRegressor(
            max_depth=2,
            min_samples_leaf=40,
            random_state=42
        )
        self.is_fitted = False
        self.bin_in_num = True

        self.num = None


    def process(self, X, fit=False):
        """
        Splits the data into categorical/numerical and applies
        any extra pre-processing before training or predicting
        """
        ###################################################################
        # NOTE: Due to current limitations, this method is specific to our
        # dataset and would need changes to work on other variations
        ################################################################
        # Categorical indicides = [0, 20, 22, 23, 27, 28, 31]
        # Binary/boolean indices = [18, 19, 21, 29, 30]

        cat_idx = np.array([0, 20, 22, 23, 27, 28, 31])
        bin_idx = np.array([18, 19, 21, 29, 30])
        both_idx = np.concatenate((cat_idx, bin_idx))

        num_X = np.delete(X, both_idx, axis=1)
        if fit:
            self.poly.fit(num_X)
        num_X = self.poly.transform(num_X)

        if fit:
            self.ohe.fit(X[:, cat_idx])
        cat_X = self.ohe.transform(X[:, cat_idx])

        if self.bin_in_num:
            num_X = np.concatenate((X[:, bin_idx], num_X), axis=1)
        else:
            cat_X = np.concatenate((X[:, bin_idx], cat_X), axis=1)

        return num_X, cat_X


    def fit(self, X_train, y_train):
        X_num, X_cat = self.process(X_train, fit=True)

        self.linreg.fit(X_num, y_train)

        y_residual = y_train - self.linreg.predict(X_num)
        self.tree.fit(X_cat, y_residual)

        self.is_fitted = True


    def predict(self, X_test):
        if not self.is_fitted:
            raise Exception("The model has not been fitted yet!")

        X_num, X_cat = self.process(X_test)

        self.num = X_num

        linreg_predict = self.linreg.predict(X_num)
        tree_predict = self.tree.predict(X_cat)
        return linreg_predict + tree_predict
    

    def evaluate_contributions(self):
        if not self.is_fitted:
            raise Exception("The model has not been fitted yet!")

        import matplotlib.pyplot as plt

        cat_feature_names = ["AccidentType", "DominantInjury", "VehicleType", "WeatherConditions",
                             "AccidentDescription", "InjuryDescription", "Gender"]
        bin_feature_names = ['ExceptionalCircumstances', 'MinorPsychologicalInjury', 'Whiplash', 'PoliceReportFiled', 'WitnessPresent']
        num_feature_names = ['SpecialHealthExpenses', 'SpecialReduction', 'SpecialOverage', 'GeneralRest', 'SpecialAdditionalInjury',
            'SpecialEarningsLoss', 'SpecialUsageLoss', 'SpecialMedications', 'SpecialAssetDamage', 'SpecialRehabilitation', 'SpecialFixes',
            'GeneralFixed', 'GeneralUplift', 'SpecialLoanerVehicle', 'SpecialTripCosts', 'SpecialJourneyExpenses', 'SpecialTherapy',
            'VehicleAge', 'DriverAge', 'NumberOfPassengers', 'AccidentDateYear', 'ClaimDateYear', 'AccidentClaimDeltaInDays', 
            'InjuryPrognosisInDays', 'AccidentDateMonthSine', 'AccidentDateMonthCosine', 'AccidentDateDaySine', 'AccidentDateDayCosine',
            'AccidentDateHourSine','AccidentDateHourCosine', 'ClaimDateMonthSine','ClaimDateMonthCosine', 'ClaimDateDaySine', 'ClaimDateDayCosine',
            'ClaimDateHourSine', 'ClaimDateHourCosine', 'PrognosisEndDateYear', 'PrognosisEndDateMonthSine', 'PrognosisEndDateMonthCosine',
            'PrognosisEndDateDaySine', 'PrognosisEndDateDayCosine', 'PrognosisEndDateHourSine', 'PrognosisEndDateHourCosine']
        poly_feature_names = self.poly.get_feature_names_out(input_features=num_feature_names)

        if self.bin_in_num:
            all_feature_names = bin_feature_names + list(poly_feature_names)
        else:
            cat_feature_names = bin_feature_names + cat_feature_names
            all_feature_names = list(poly_feature_names)

        # Now get coefficients and make sure they match amount of feature names
        coefs = self.linreg.coef_
        if len(coefs) != len(all_feature_names):
            raise Exception(f"Mismatch between number of coefficients ({len(coefs)}) and features ({len(all_feature_names)})!")

        # Pair names and coefficients, and then sort by absolute value
        feature_coef_pairs = list(zip(all_feature_names, coefs))
        sorted_pairs = sorted(feature_coef_pairs, key=lambda x: abs(x[1]), reverse=True)

        print("\nTop Features by Contribution (Linear Regression):")
        for name, coef in sorted_pairs:
            print(f"{name:28s}| {coef:.6f}")

        # Again
        feature_coef_pairs = list(zip(cat_feature_names, self.tree.feature_importances_))
        sorted_pairs = sorted(feature_coef_pairs, key=lambda x: abs(x[1]), reverse=True)

        print("\nTop Features by Contribution (Residual Decision Tree):")
        for name, coef in sorted_pairs:
            print(f"{name:28s}| {coef:.6f}")

        # Do shap now
        explainer_lin = shap.LinearExplainer(self.linreg, self.num)
        shap_values_lin = explainer_lin.shap_values(self.num)
        shap.summary_plot(shap_values_lin, self.num, feature_names=all_feature_names, max_display=20)
