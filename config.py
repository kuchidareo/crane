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
}

position_to_sensor = {}
for key, val in sensor_to_position.items():
    if val in position_to_sensor:
        position_to_sensor[val].append(key)
    else:
        position_to_sensor[val] = [key]

position_to_label = {
    value.RIGHT_ARM: "LL_Right_Arm",
    value.LEFT_ARM: "LL_Left_Arm",
    value.LOCOMOTION: "Locomotion"
}