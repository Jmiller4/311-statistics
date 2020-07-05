# 311-statistics
Statistics on 311 response times, by request type and demographic.


Currently, fairness_calculations.py has a method to return the average response time for a given demographic group, for a given type of 311 request. Calling the method requires understanding the codes used to denote requests and demographics. Mappings between codes and human-readable demographic names/ request types are given in "service_request_short_codes.csv", and "code-descriptions-*.csv" in the "tables" folder.

The other python files were used to prepare the tables "bg_averages.csv" and "demographics_table.csv". To understand these files, read them in this order:
1. bg_decider.py
2. bg_appender.py
3. call_type_splitter.py
4. bg_reformatter.py
5. response_time_aggregator.py
6. demographic_table_creator.py

This may look like a lot of files, but each one is quite short. Also note that you'll need to download some data from the elsewhere if you want to run any of these files (details in the individual files). And be warned that "bg_appender" takes quite a while to run. 

Some request types have not made their way into this dataset:

- 311 information only calls, and aircraft noise complaints, have immediate responses and so are not considered.

- Requests with code FAC (Fire Safety Inspection Request) have no completion times in the dataset, so they are excluded as well.