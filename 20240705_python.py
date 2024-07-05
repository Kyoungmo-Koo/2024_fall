import math
import serial
import csv

MAX_POS = 65535
ARRAY_SIZE = 5000
SEND_SIZE = 2500
UART_PORT = 'COM4'
BAUD_RATE = 5000000

count = 0

def pack_position_data(x_pos, y_pos):
    return ((x_pos & 0xFFFF) << 16) | (y_pos & 0xFFFF)

def generate_sine_position_data(array_size):
    position_data_array = []
    for i in range(array_size):
        x = int(MAX_POS * math.sin(1.0 * math.pi * i / array_size)) & 0xFFFF
        y = int(MAX_POS * math.sin(1.0 * math.pi * i / array_size)) & 0xFFFF
        position_data_array.append(pack_position_data(x, y))
    return position_data_array

def generate_step_position_data(step, array_size):
    position_data_array = []
    for i in range(1, array_size + 1):
        x = step * i
        y = step * i
        position_data_array.append(pack_position_data(x, y))
    return position_data_array

sine_position_data = generate_sine_position_data(ARRAY_SIZE)
step_position_data = generate_step_position_data(6, ARRAY_SIZE)


# UART setup
ser = serial.Serial(port=UART_PORT,\
    baudrate=BAUD_RATE,\
    parity=serial.PARITY_NONE,\
    stopbits=serial.STOPBITS_ONE,\
    bytesize=serial.EIGHTBITS,\
        timeout=0)
ser.reset_input_buffer()
ser.reset_output_buffer()

import time
import struct
start_time = 0
end_time = 0
count = 0

binary_data1 = b''.join(struct.pack('<I', data) for data in step_position_data[0: 2500])
binary_data2 = b''.join(struct.pack('<I', data) for data in step_position_data[2500:5000])
binary_data3 = b''.join(struct.pack('<I', data) for data in sine_position_data[0:2500])
binary_data4 = b''.join(struct.pack('<I', data) for data in sine_position_data[2500:5000])

csvfile = open('output.csv', 'a', newline='')
csvwriter = csv.writer(csvfile)

try:
    while True:
        #time.sleep(1)
        if ser.in_waiting > 0:
            start_time = time.time()
            print(start_time - end_time)
            #received_data = ser.readline().lower()
            #value = int.from_bytes(received_data, byteorder='big', signed=False)
            if(count % 4 == 0):
                data = ser.read(4000)
                #values = [int.from_bytes(data[i:i+1], byteorder='big', signed=False) for i in range(100)]
                #csvwriter.writerow(values)
                #time.sleep(0.001)
                ser.write(binary_data1)
            elif(count % 4 == 1):
                data = ser.read(4000)
                #values = [int.from_bytes(data[i:i+1], byteorder='big', signed=False) for i in range(100)]
                # csvwriter.writerow(values)
                #time.sleep(0.001)
                ser.write(binary_data2)
            elif(count % 4 == 2):
                data = ser.read(4000)
                #values = [int.from_bytes(data[i:i+1], byteorder='big', signed=False) for i in range(100)]
                # csvwriter.writerow(values)
                #time.sleep(0.001)
                ser.write(binary_data3)
            elif(count % 4 == 3):
                data = ser.read(4000)
                #values = [int.from_bytes(data[i:i+1], byteorder='big', signed=False) for i in range(100)]
                # csvwriter.writerow(values)
                # time.sleep(0.001)
                ser.write(binary_data4)            
            
            count = count + 1
            ser.reset_input_buffer()
            # if (value == 0):
            #     print(value)
            #     ser.write(binary_data1)
                #print(time.time() - start_time) 
            # elif (value == 1):
            #     print(value)
            #     ser.write(binary_data2)
                #print(time.time() - start_time) 
            # elif (value == 2):
            #     print(value)
            #     ser.write(binary_data3)
                #print(time.time() - start_time) 
            # elif (value == 3):
            #     print(value)
            #     ser.write(binary_data4)
                #print(time.time() - start_time) 

            end_time = time.time()
            print(end_time - start_time)

except KeyboardInterrupt:
    print("\nKeyboard Interrupt")
finally:
    ser.close()
