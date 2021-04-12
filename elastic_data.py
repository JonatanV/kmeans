import json
import datetime
import time

with open(r'C:\code\kmeans\xploreelasticutv-1000.json ', 'r') as f:
    data = json.load(f)

timestamps = []
for line in data:
    timestamp = ""

    for i in range(24):
        timestamp += str(line)[139 + i]
    utc_timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%fZ')
    unix_timestamp = time.mktime(utc_timestamp.timetuple())
    timestamps.append(str(unix_timestamp))
    # timestamps = sorted(timestamps, key=lambda x: float(x))
    # print(timestamps)
with open("test.txt", "a") as d:
    d.write("\n".join(timestamps))
    d = open("test.txt", "r")
    d.close()
