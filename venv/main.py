from vnstock import listing_companies, company_profile,company_overview
import os
import json

companies = listing_companies()

print(companies.count())

def saveCompanyOverview(data_dict,filename="comapnyOverviews.json"):
    
    # Check if file exists, if not, create an empty list in it
    if not os.path.exists(filename):
        with open(filename, 'w') as json_file:
            json.dump([], json_file,indent=4)

    try:
        with open(filename, 'r') as json_file:
            current_data = json.load(json_file)
    except json.decoder.JSONDecodeError:
        # Handle the error, for example, by initializing current_data as an empty list
        current_data = []

    # Append the new data dictionary
    current_data.append(data_dict)

    # Write updated data back to the file
    with open(filename, 'w') as json_file:
        json.dump(current_data, json_file, indent=4)

def getCompanyInfo():
    #top1 = companies.iloc[0:1]
    i = 1
    for indexCompany,company in companies.iterrows():
        symbol = company['ticker']
        try:
            companyProfile = company_overview(symbol)
            for indexOverview,companyOverview in companyProfile.iterrows():
                    if companyOverview["ticker"]:
                        saveCompanyOverview({
                            "index" : i,
                            "ticker" : companyOverview["ticker"],
                            "exchange" : companyOverview["exchange"],
                            "companyType" : companyOverview["companyType"],
                            "industry" : companyOverview["industry"],  
                            "industryID" : companyOverview["industryID"],
                            "industryEn" : companyOverview["industryEn"],
                            "shortName" : companyOverview["shortName"],
                        })
                        i+= 1
                #print(companyOverview)
        except :
           print("Error")

      

getCompanyInfo()

