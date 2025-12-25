# chem-calc-tools.py
# Python 3.10.5

import urllib2
import urllib
import urllib.request
import urllib.parse
import json

ccurl = 'https://www.chemcalc.org/chemcalc/mf'

# Define a molecular formula string
mf = 'C100H100'

# Define the parameters and send them to Chemcalc
params = {
    'mf': mf,
    'isotopomers': 'jcamp,xy'
}
response = urllib.request.urlopen(
    ccurl, urllib.parse.urlencode(params).encode('utf-8'))

# Read the output and convert it from JSON into a Python dictionary
jsondata = response.read()
data = json.loads(jsondata.decode('utf-8'))

print(data)

# Find possible molecular formula for a specific monoisotopic mass

chemcalcURL = 'https://www.chemcalc.org/chemcalc/em'

# Define a molecular formula string
mfRange = 'C0-100H0-100N0-10O0-10'

# target mass
mass = 300

# Define the parameters and send them to Chemcalc
# other options(mass tolerance, unsaturation, etc.params = {
    'mfRange': mfRange,
    'monoisotopicMass': mass
}
# Read the output and convert it from JSON into a Python dictionary jsondata = response.read()data = json.loads(jsondata)print data
response = urllib2.urlopen(chemcalcURL, urllib.urlencode(params))
