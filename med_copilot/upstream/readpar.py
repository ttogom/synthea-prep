import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

dic='./guidelines/output/'

entities_path=dic+'entities.parquet'
entities=pd.read_parquet(entities_path)

print(entities.shape)
print(entities.head(10))
print('\n')
# print(entities['type'].value_counts())

community_report_path=dic+'community_reports.parquet'
community_report=pd.read_parquet(community_report_path)

print(community_report.shape)
print(community_report.head(10))

relationships_path=dic+'relationships.parquet'
relationships=pd.read_parquet(relationships_path)

print(relationships.shape)
print(relationships.head(10))