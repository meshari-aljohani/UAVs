#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 10:34:41 2023

@author: mesh
"""

import csv
import random

random.seed(4)

header_base = ["Latitude", "Longitude", "Altitude", "Temperature", "Wind_Speed", "Humidity", "Acceleration", "Battery_Level", "Obstacle_Detected"]
task_types = ["takeoff", "search", "Target_Detected", "attack", "Elimination_Proof", "land"]

header = ["Mission_ID"]
for task_type in task_types:
    header += [f"{task_type}_{col}" for col in header_base]

header.append("Success")
messionNo = range(0,50,1)
for mis in messionNo:
    random.seed(4)
    with open("F"+str(mis)+".csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for mission in range(mis+1):
            row = [mission + 1]
            for task_id, task_type in enumerate(task_types, start=1):
                latitude = random.uniform(-90, 90)
                longitude = random.uniform(-180, 180)

                altitude_range = (0, 100) if task_id == 1 else (2000, 3000)
                altitude = random.uniform(*altitude_range)

                temperature = random.uniform(15, 30)
                wind_speed = random.uniform(0, 5)
                humidity = random.uniform(30, 60)
                acceleration_range = (1, 3) if task_id == 1 else (0, 2)
                acceleration = random.uniform(*acceleration_range)

                battery_level = [100, 95, 90, 80, 75, 70, 65][task_id - 1]
                obstacle_detected = random.uniform(0, 0.3)

                temperature = temperature + (temperature * random.random()*0.2)
                wind_speed = wind_speed + (wind_speed * random.random()*0.2)
                humidity = humidity + (humidity * random.random()*0.2)
                acceleration = acceleration + (acceleration * random.random()*0.2)
                obstacle_detected = obstacle_detected + (obstacle_detected * random.random()*0.2)

                task_data = [latitude, longitude, altitude, temperature, wind_speed, humidity,  acceleration, battery_level, obstacle_detected]
                row.extend(task_data)

            row.append(1)  # Success at the end of each mission
            writer.writerow(row)

        for mission in range(mis+1):
            row = [mission + 1]
            for task_id, task_type in enumerate(task_types, start=1):

                latitude = random.uniform(-90, 90)
                longitude = random.uniform(-180, 180)

                altitude_range = (0, 100) if task_id == 1 else (2000, 3000)
                altitude = random.uniform(*altitude_range)

                temperature = random.uniform(15, 30)
                wind_speed = random.uniform(0, 5)
                humidity = random.uniform(30, 60)
                acceleration_range = (1, 3) if task_id == 1 else (0, 2)
                acceleration = random.uniform(*acceleration_range)

                battery_level = [100, 95, 90, 80, 75, 70, 65][task_id - 1]
                obstacle_detected = random.uniform(0, 0.3)

                temperature = temperature + (temperature * random.random())
                wind_speed = wind_speed + (wind_speed * random.random())
                humidity = humidity + (humidity * random.random())
                acceleration = acceleration + (acceleration * random.random())
                obstacle_detected = obstacle_detected + (obstacle_detected * random.random())

                task_data = [latitude, longitude, altitude, temperature, wind_speed, humidity, acceleration, battery_level, obstacle_detected]
                row.extend(task_data)

            row.append(0)  # Failure at the end of each abnormal mission
            writer.writerow(row)
