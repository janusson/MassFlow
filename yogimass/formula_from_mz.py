# usr/bin/python3
#  -*- coding: utf-8 -*-
# Description: Given a m/z value, returns a list of molecular formulae matching the m/z value.
# ? Currently implemented as manual input, but can be automated for larger datasets
import requests
import pandas as pd


def mf_from_accmass(
    mz, mass_tolerance=0.5, formula_range="C0-100H0-202N0-10O0-10F0-3Cl0-3Br0-1(+)"
):
    """
    Returns molecular formulae matches for a given accurate m/z.
    Returns a DataFrame of ChemCalc search results sorted by absolute value of error.
    Prints a table of the top 10 most accurate molecular formulas.
    ms (float): m/z value
    options (dict): Options for chemcalc search
    """
    chemcalcURL = "https://www.chemcalc.org/chemcalc/em"
    options = {
        "mfRange": formula_range,
        "numberOfResultsOnly": False,
        "typedResult": False,
        "useUnsaturation": False,
        "minUnsaturation": 0,
        "maxUnsaturation": 50,
        "jcampBaseURL": "http://www.chemcalc.org/service/jcamp/",
        "monoisotopicMass": mz,
        "jcampLink": True,
        # The 'jcamplink' returns a link to a file containing isotopic
        # distribution of retrieved molecular structure.
        "integerUnsaturation": False,
        # Ions/Radicals can have non-integer unsaturation
        "referenceVersion": "2013",
        "massRange": mass_tolerance,
        #              'minMass': -0.5,
        #              'maxMass': 0.5,
    }
    response = requests.get(chemcalcURL, options).json()

    df1 = pd.DataFrame(response["results"])
    df1["abs_error"] = df1["error"].abs()
    df1.sort_values(by="abs_error", ascending=True, inplace=True)
    print(f"> > > Top ten hits for {mz}:\n {df1.head(10)}")

    return df1


x = input("Enter m/z: ")
result_of_search = mf_from_accmass(
    mz=x,
    mass_tolerance=0.5,
    # 'mfRange': 'C0-100H0-202N0-10O0-10F0-3Cl0-3Br0-1(+)'
    formula_range="C6-100H0-100O1-10N0-3(+)",
)
