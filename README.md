# 311-statistics
The main product of this work (so far) are the files "avg_times_by_demo_and_type.csv" and "avg_queue_cuts_*.csv". The former gives estimated wait times by demographic and request type, the latter files measure the extent to which requests are de-prioiritized, by demographic and request type, and with varying levels of strictness. 

The "avg_queue_cuts" files measure the extent to which 311 requests are answered on a first-in, first-out basis. Row i, column j of avg_queue_cuts_X.csv gives the average number of requests that will be answered before a request of type j from a member of demographic i, despite coming in after that request, and specifically in an X-long period after that request.   

A more in-depth explanation of the methodology can be found here:

https://www.notion.so/311-Statistics-Methodology-46037e1723914c4097db349fd0ca2c40 

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