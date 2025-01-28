import value

sensor_to_position = {
    "RUA": value.RIGHT_ARM,
    "RLA": value.RIGHT_ARM,
    "LUA": value.LEFT_ARM,
    "LLA": value.LEFT_ARM,
    "R-SHOE": value.LOCOMOTION,
    "L-SHOE": value.LOCOMOTION,
    "RUA^": value.RIGHT_ARM,
    "RUA_": value.RIGHT_ARM,
    "RWR": value.RIGHT_ARM,
    "RH": value.RIGHT_ARM,
    "LUA^": value.LEFT_ARM,
    "LUA_": value.LEFT_ARM,
    "LWR": value.LEFT_ARM,
    "LH": value.LEFT_ARM,
    "RKN^": value.LOCOMOTION,
    "RKN_": value.LOCOMOTION,
    "CUP": value.OBJECTS,
    "SALAMI": value.OBJECTS,
    "WATER": value.OBJECTS,
    "CHEESE": value.OBJECTS,
    "BREAD": value.OBJECTS,
    "KNIFE1": value.OBJECTS,
    "MILK": value.OBJECTS,
    "SPOON": value.OBJECTS,
    "SUGAR": value.OBJECTS,
    "KNIFE2": value.OBJECTS,
    "PLATE": value.OBJECTS,
    "GLASS": value.OBJECTS,
    "DOOR1": value.OBJECTS,
    "LAZYCHAIR": value.OBJECTS,
    "DOOR2": value.OBJECTS,
    "DISHWASHER": value.OBJECTS,
    "UPPERDRAWER": value.OBJECTS,
    "LOWERDRAWER": value.OBJECTS,
    "MIDDLEDRAWER": value.OBJECTS,
    "FRIDGE": value.OBJECTS,
}

high_level_position_to_position_mapping = {
    value.BOTH_ARMS: [value.RIGHT_ARM, value.LEFT_ARM, value.OBJECTS],
    value.LEFT_OBJECT: [value.OBJECTS, value.LEFT_ARM],
    value.RIGHT_OBJECT: [value.OBJECTS, value.RIGHT_ARM],
}

position_to_sensor = {}
for key, val in sensor_to_position.items():
    if val in position_to_sensor:
        position_to_sensor[val].append(key)
    else:
        position_to_sensor[val] = [key]

for key, val in high_level_position_to_position_mapping.items():
    sensors = []
    for position in val:
        sensors.extend(position_to_sensor[position])
    position_to_sensor[key] = sensors


position_to_original_position_label = {
    value.RIGHT_ARM: "LL_Right_Arm",
    value.LEFT_ARM: "LL_Left_Arm",
    value.LOCOMOTION: "Locomotion",
    value.LEFT_OBJECT: "LL_Left_Arm_Object",
    value.RIGHT_OBJECT: "LL_Right_Arm_Object",
    value.BOTH_ARMS: "ML_Both_Arms",
}