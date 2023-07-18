import pandas as pd
from radarpipeline.datalib import RadarData
from radarpipeline.features import Feature, FeatureGroup
import pyspark.pandas as ps


class PassiveDataFeatures(FeatureGroup):
    def __init__(self):
        name = "PassiveDataFeatures"
        description = "contains Passive Data Features"
        features = [ActiveSessions, NotificationResponseLatency, AmbientLightActive]
        super().__init__(name, description, features)

    def preprocess(self, data: RadarData) -> RadarData:
        """
        Preprocess the data for each feature in the group.
        """
        return data

class ActiveSessions(Feature):
    def __init__(self):
        self.name = "ActiveSessions"
        self.description = "The duration of active sessions"
        self.required_input_data = ["android_phone_user_interaction"]

    def preprocess(self, data: RadarData) -> RadarData:
        df_user_interaction = data.get_combined_data_by_variable(
            "android_phone_user_interaction"
        )
        df_user_interaction['value.time'] = pd.to_datetime(df_user_interaction['value.time'], unit="s")
        df_user_interaction['value.timeReceived'] = pd.to_datetime(df_user_interaction['value.timeReceived'], unit="s")
        df_user_interaction['key.userId'] = df_user_interaction['key.userId'].str.strip()
        # Removing duplicates
        df_user_interaction = df_user_interaction[~df_user_interaction[["key.userId", "value.time", "value.interactionState"]].duplicated()]
        df_user_interaction.reset_index(drop=True, inplace=True)
        df_user_interaction["previous_interactionState"] = df_user_interaction.groupby("key.userId")['value.interactionState'].shift()
        df_user_interaction["previous_interactionState_time"] = df_user_interaction.groupby("key.userId")['value.time'].shift()
        df_unlocked_duration = df_user_interaction[df_user_interaction["previous_interactionState"]=="UNLOCKED"].reset_index(drop=True)
        df_unlocked_duration = df_unlocked_duration.rename({"key.userId":"uid", "value.time":"time", "value.interactionState":"interactionState"}, axis=1)
        df_unlocked_duration["begin_time"] = df_unlocked_duration["previous_interactionState_time"].astype('datetime64[s]')
        df_unlocked_duration["end_time"] = df_unlocked_duration["time"].astype('datetime64[s]')
        df_active_session_details = df_unlocked_duration[['key.projectId', 'uid', 'key.sourceId',
       'begin_time', 'end_time']]
        data._cache = {}
        data._cache['df_active_session_details'] = df_active_session_details
        return df_active_session_details

    def calculate(self, data) -> float:
        """
        Calculate the feature.
        """
        return data

class NotificationResponseLatency(Feature):
    def __init__(self):
        self.name = "NotificationResponseLatency"
        self.description = "The duration between notification and response"
        self.required_input_data = ["android_phone_user_interaction", "android_phone_usage_event"]

    def _get_nearest_unlocked_time(self, row):
        uid = row['key.userId'].unique()[0]
        df_temp = self.df_user_interaction_unlocked[self.df_user_interaction_unlocked['key.userId'] == uid]
        time_arr = []
        df_temp_arr = df_temp["value.time"].tolist()
        t_pointer = 0
        for t in row["value.time"]:
            while t > df_temp_arr[t_pointer]:
                t_pointer += 1
                if t_pointer == len(df_temp_arr):
                    break
            if t_pointer == len(df_temp_arr):
                    break
            time_arr.append(df_temp_arr[t_pointer])
        time_arr +=  [None] * (len(row["value.time"]) - len(time_arr))
        row["unlocked_time"]  =    time_arr
        return row

    def preprocess(self, data: RadarData) -> RadarData:
        df_user_interaction = data.get_combined_data_by_variable(
            "android_phone_user_interaction"
        )
        df_phone_usage = data.get_combined_data_by_variable(
            "android_phone_usage_event"
        )
        df_user_interaction['value.time'] = pd.to_datetime(df_user_interaction['value.time'], unit="s")
        df_user_interaction['value.timeReceived'] = pd.to_datetime(df_user_interaction['value.timeReceived'], unit="s")
        df_user_interaction['key.userId'] = df_user_interaction['key.userId'].str.strip()
        # Removing duplicates
        df_user_interaction = df_user_interaction[~df_user_interaction[["key.userId", "value.time", "value.interactionState"]].duplicated()]
        df_user_interaction.reset_index(drop=True, inplace=True)
        self.df_user_interaction_unlocked = df_user_interaction[df_user_interaction["value.interactionState"] == "UNLOCKED"].reset_index(drop=True)
        # Cleaining phone usage data
        df_phone_usage['value.time'] = pd.to_datetime(df_phone_usage['value.time'], unit="s")
        df_phone_usage['value.timeReceived'] = pd.to_datetime(df_phone_usage['value.timeReceived'], unit="s")
        df_phone_usage['key.userId'] = df_phone_usage['key.userId'].str.strip()
        df_phone_usage_foreground = df_phone_usage[df_phone_usage["value.eventType"] == "FOREGROUND"]
        df_user_interaction = df_user_interaction.sort_values(["key.userId", "value.time"]).reset_index(drop=True)
        return df_phone_usage_foreground

    def calculate(self, data) -> float:
        """
        Calculate the feature.
        """
        df_phone_usage_foreground = data
        df_phone_usage_foreground = df_phone_usage_foreground.groupby("key.userId").apply(self._get_nearest_unlocked_time)
        df_phone_usage_foreground["unlocked_timediff"] = df_phone_usage_foreground["unlocked_time"] - df_phone_usage_foreground["value.time"]
        df_phone_usage_foreground["unlocked_timediff_sec"] = df_phone_usage_foreground["unlocked_timediff"].dt.seconds
        return df_phone_usage_foreground

class AmbientLightActive(Feature):
    def __init__(self):
        self.name = "AmbientLightActive"
        self.description = "Ambient light when participants are actively using their phones"
        self.required_input_data = ["android_phone_light"]

    def _get_ambient_light_data(self, row):
        uid, begin_time, end_time = row['uid'], row['begin_time'], row['end_time']
        temp_df = self.df_light_minute[(self.df_light_minute ['uid']==uid) & (self.df_light_minute ['time_mins']>=begin_time) & (self.df_light_minute ['time_mins'] < end_time)]
        temp_df['unlock_begin_time'] = begin_time
        temp_df['unlock_end_time'] = end_time
        return temp_df

    def preprocess(self, data: RadarData) -> RadarData:
        df_light = data.get_combined_data_by_variable(
            "android_phone_light"
        )
        df_active_session_details = data._cache['df_active_session_details']
        df_light['value.time'] = pd.to_datetime(df_light['value.time'], unit="s")
        df_light['value.timeReceived'] = pd.to_datetime(df_light['value.timeReceived'], unit="s")
        df_light['key.userId'] = df_light['key.userId'].str.strip()
        df_light["time_mins"] = df_light["value.time"].astype('datetime64[m]')
        df_light = df_light.rename({"value.light" : "light", "key.userId": "uid", "value.time.hour":"hour"}, axis=1)
        df_light_minute = df_light.groupby(["uid", "time_mins"]).agg({"light": "mean"}).reset_index()
        self.df_light_minute = df_light_minute.dropna().reset_index(drop=True)
        return df_active_session_details

    def calculate(self, data) -> float:
        """
        Calculate the feature.
        """
        df_active_session_details = data
        df_unlocked_light = df_active_session_details[['uid', 'begin_time', 'end_time']].apply(self._get_ambient_light_data, axis=1)
        df_unlocked_light_concated = pd.concat(df_unlocked_light.to_list())
        return df_unlocked_light_concated