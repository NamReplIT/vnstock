from vnstock import listing_companies, company_profile,company_overview
import requests

companies = listing_companies()

def getCompanyInfo():
    top10 = companies.iloc[0:10]
    for indexCompany,company in top10.iterrows():
       symbol = company['ticker']
       companyProfile = company_overview(symbol)
       for indexOverview,companyOverview in companyProfile.iterrows():
           print(companyOverview['industryEn'])

      

getCompanyInfo()