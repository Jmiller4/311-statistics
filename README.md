# 311-statistics


**Estimating Bucket Times, and what the bucket columns mean**

One of the things we're interested in is examining the extent to which the requests are completed on a first-in, first-out basis. So we could compare the list of requests sorted by creation time with that same list sorted by completion time, and see what's out of order. But we also might want to have some leniency for requests close to each other: for example, if request A comes in at 1:05PM and request B comes in at 1:10PM, maybe it's fine if B is completed before A. The bucket columns are essentially one way of allowing for that leniency.

 - CR_BUCKET_X for X in {1SEC, 1HR, 3HR, 6HR, 24HR}: these columns describe the X-long period in which the request was created. To calculate these numbers, I chose a date right before the earliest record in any dataset. Then I found the difference in seconds between each row's creation date and midnight on that date -- that gave me CR_BUCKET_1SEC. Then, to get CR_BUCKET_1HR, I divided CR_BUCKET_1SEC by 3600 (the number of seconds in an hour) and rounded down. To get CR_BUCKET_3HR, I divided CR_BUCKET_1SEC by 3600 * 3 and rounded down, and so forth. So, if request A was created at 1:45 and request B was created at 2:07, they would have different values for CR_BUCKET_1HR (and specifically B's value would be 1 greater than A's), but they would have the same value for CR_BUCKET_3HR, 6HR, and so on.
    CR_BUCKET_X_DISPLACEMENT for X in {1SEC, 1HR, 3HR, 6HR, 24HR}: This column represents how early or late the request was completed, compared to when it would be completed if everything was perfectly first-in, first-out, and allowing for leniency with regards to buckets of size X. A negative value means the request was completed earlier than it should have been completed, and a positive value means that the request was completed after it should have been completed. Just to be super clear, here's an example. Suppose that 10 requests come in between 3PM and 4PM, and another 10 requests come in between 4PM and 5PM. Now, in a world where everything was first-in, first-out, we'd expect to see all the 3-4PM requests completed first, and then all the 4-5PM requests completed after. But, let's say that in our dataset, there was a 3-4PM request that was completed 15th. Well, in a first-in first-out world, the latest this request would possibly be scheduled was 10th. So we give it a displacement of 15-10 = 5. On the other hand, say that a 4-5PM request was completed 2nd. In a FIFO world, the earliest this request should be completed is 10th. So we give it a displacement of 2-10 = -8. Lastly, suppose a 3-4PM request was completed 7th. This fits within when we'd expect this request to be completed in a FIFO world (between 1st and 10th), so we give it a displacement of 0. We would do the same thing for a 4-5PM request completed, say, 13th

**Other Notes**

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