# Passive data Pipeline

Integrating code for passive data pipeline about phone usage activity.

# Description

Passive data pipeline is a set of scripts that process the data collected from the phone usage activity.

## Data
The data is stored as .csv.gz. format, which the I/O module reads and convert into a Spark DataFrame for further processing.

## Features
### ActiveSessions

The duration of active sessions in the phone.
### NotificationResponseLatency

The time between the notification is received and the user unlock the device.

### AmbientLightActive

The ambient light in the phone during the active sessions.
## Output

The output is 3 csv files `active_sessions.csv`, `notification_response_latency.csv` and `ambient_light_active.csv`.