# 311-statistics
The main product of this work (so far) is the file "avg_times_by_demo_and_type.csv", which lists average 311 response times for different demographics (rows) and types of 311 requests (columns). "311_stats_methodology.pdf" details my approach to creating that table and "fairness_calculations.py" is the code that produced that table.

A few python files were used to prepare the tables "bg_averages.csv" and "demographics_table.csv", which were in turn used to calculate the "average response times..." file. To understand how the python files were used, read them in this order:
1. bg_decider.py
2. bg_appender.py
3. call_type_splitter.py
4. bg_reformatter.py
5. response_time_aggregator.py
6. demographic_table_creator.py

They're all in the "data preparation" folder. Each one is quite short. Also note that you'll need to download some data from the elsewhere if you want to run any of these files (details in the individual files). And be warned that "bg_appender" takes quite a while to run. 

Some request types have not made their way into this dataset:

- 311 information only calls and aircraft noise complaints have immediate responses and so are not considered.

- Requests with code FAC (Fire Safety Inspection Request) have no completion times in the dataset, so they are excluded as well.