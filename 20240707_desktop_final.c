//Red Wire = 3V3
//Black Wire = Ground
//Orange Wire = SCLK
//Yellow Wire = MOSI
//Green Wire = MISO
//Brown Wire = CS

#include <chrono>
#include <vector>
#include <random>
#include <climits>
#include <algorithm>
#include <functional>

#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <Windows.h>
#include <time.h>
#include <math.h>
#include "ftd2xx.h"
#include "libmpsse_spi.h"

HANDLE hComm;

void print_and_quit(const char cstring[]) {
    printf("%s\n", cstring);
    getc(stdin);
    exit(1);
}
void Init_UART() {
    BOOL status;
    DCB dcbSerialParams = { 0 };
    COMMTIMEOUTS timeouts = { 0 };

    hComm = CreateFile(L"\\\\.\\COM6", GENERIC_READ | GENERIC_WRITE, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hComm == INVALID_HANDLE_VALUE) {
        printf("Error in opening serial port\n");
    }
    else {
        printf("Opening serial port successful\n");
    }
    status = PurgeComm(hComm, PURGE_RXCLEAR | PURGE_TXCLEAR);
    if (!status) {
        printf("Error in purging comm\n");
        CloseHandle(hComm);
    }
    status = GetCommState(hComm, &dcbSerialParams);
    if (!status) {
        printf("Error in GetCommState\n");
        CloseHandle(hComm);
    }

    dcbSerialParams.BaudRate = CBR_9600;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;
    status = SetCommState(hComm, &dcbSerialParams);

    if (!status) {
        printf("Error in setting DCB structure\n");
        CloseHandle(hComm);
    }

    timeouts.ReadIntervalTimeout = 200000;
    timeouts.ReadTotalTimeoutMultiplier = 0;
    timeouts.ReadTotalTimeoutConstant = 200000;
    timeouts.WriteTotalTimeoutMultiplier = 0;
    timeouts.WriteTotalTimeoutConstant = 200000;
    if (!SetCommTimeouts(hComm, &timeouts)) {
        printf("Error in setting timeouts\n");
        CloseHandle(hComm);
    }
}
FT_HANDLE Init_SPI() {
    FT_STATUS status;
    FT_DEVICE_LIST_INFO_NODE channelInfo;
    FT_HANDLE handle;

    DWORD channelCount = 0;
    status = SPI_GetNumChannels(&channelCount);

    if (status != FT_OK)
        print_and_quit("Error while checking the number of available MPSSE channels.");
    else if (channelCount < 1)
        print_and_quit("Error: No MPSSE channels are available.");

    printf("There are %d channels available. \n\n", channelCount);

    for (int i = 0; i < channelCount; i++) {
        status = SPI_GetChannelInfo(i, &channelInfo);
        if (status != FT_OK)
            print_and_quit("Error while getting details for an MPSSE channel.");
        printf("Channel number : %d\n", i);
        printf("Description: %s\n", channelInfo.Description);
        printf("Serial Number : %d\n", channelInfo.SerialNumber);
    }

    uint32_t channel = 0;
    printf("\n Enter a channel number to use: ");
    scanf_s("%d", &channel);

    status = SPI_OpenChannel(channel, &handle);
    if (status != FT_OK)
        print_and_quit("Error while opening the MPSSE channel.");

    ChannelConfig channelConfig;
    channelConfig.ClockRate = 15000000;
    channelConfig.configOptions = SPI_CONFIG_OPTION_MODE2 | SPI_CONFIG_OPTION_CS_DBUS3 | SPI_CONFIG_OPTION_CS_ACTIVELOW;
    channelConfig.LatencyTimer = 20;

    status = SPI_InitChannel(handle, &channelConfig);
    if (status != FT_OK)
        print_and_quit("Error while initializing the MPSSE channel.");
    return handle;
}
void reversePacketOrder(UCHAR* buffer, size_t bufferSize) {
    for (size_t i = 0; i < bufferSize; i += 4) {
        for (size_t j = 0; j < 2; ++j) {
            UCHAR temp = buffer[i + j];
            buffer[i + j] = buffer[i + 3 - j];
            buffer[i + 3 - j] = temp;
        }
    }
}

int test(int argc, char** argv)
{
    FT_HANDLE handle;
    FILE* fp;
    FILE* fr;
    FILE* fr2;
    Init_UART();
    Init_libMPSSE();
    handle = Init_SPI();
    fopen_s(&fp, "data.csv", "w+");
    fopen_s(&fr, "EnginePattern_raster_amp3_res512_hexadecimal_downsample.txt", "r");
    fopen_s(&fr2, "EnginePattern_raster_amp3_res512_hexadecimal_downsample.txt", "r");

    FT_STATUS status;
    bool status2;
    DWORD bytesRead;
    DWORD bytesWrite;
    UCHAR StartByte = 1;
    UCHAR RxByte;

    DWORD transferCount = 0;
    LPDWORD ptransferCount = &transferCount;
    DWORD options = SPI_TRANSFER_OPTIONS_SIZE_IN_BYTES | SPI_TRANSFER_OPTIONS_CHIPSELECT_ENABLE | SPI_TRANSFER_OPTIONS_CHIPSELECT_DISABLE;
    //Write how many trials user want
    int COUNT = 320;
    //Write how many positions user want inside a buffer
    int NUM_OF_POSITIONS = 6264 * 8;
    DWORD input_data[2 * 6264 * 8];
    //Write how many bytes user want inside a buffer (4 * NUM_OF_POSITIONS)
    UCHAR tx_buffer[2 * 4 * 6264 * 8];
    //Write how many bytes user want to transmit & receive a time (2 * NUM_OF_POSITIONS)
    int NUM_OF_BYTES_PER_CHUNK = 12528;
    UCHAR rx_buffer[12528];

    int x;
    int y;
    long position;
    int CHUNK_NUM = 0;

    status = WriteFile(hComm, &StartByte, 1, &bytesWrite, NULL);
    printf("sent");
    status = ReadFile(hComm, &RxByte, 1, &bytesRead, NULL);
    printf("%d \n", RxByte);
    for (int k = 0; k < 2 * NUM_OF_POSITIONS; ++k) {
        fscanf_s(fr, "%x", &input_data[k]);
    }
    for (int k = 0; k < 2 * NUM_OF_POSITIONS; ++k) {
        uint32_t data = input_data[k];
        tx_buffer[4 * k + 3] = (uint8_t)((data >> 24) & 0xFF);
        tx_buffer[4 * k + 2] = (uint8_t)((data >> 16) & 0xFF);
        tx_buffer[4 * k + 1] = (uint8_t)((data >> 8) & 0xFF);
        tx_buffer[4 * k] = (uint8_t)(data & 0xFF);
    }


    for (int i = 0; i < COUNT; i++) {
        Sleep(1);
        status = SPI_Read(handle, &rx_buffer[0], NUM_OF_BYTES_PER_CHUNK, ptransferCount, options);
        Sleep(1);

        //if (i % 4 == 0) {
        //    status = SPI_Write(handle, &tx_buffer[0], NUM_OF_BYTES_PER_CHUNK, ptransferCount, options);
        //}
        //else if (i % 4 == 1) {
        //    status = SPI_Write(handle, &tx_buffer[NUM_OF_BYTES_PER_CHUNK], NUM_OF_BYTES_PER_CHUNK, ptransferCount, options);
        //}
        //else if (i % 4 == 2) {
        //    status = SPI_Write(handle, &tx_buffer[2 * NUM_OF_BYTES_PER_CHUNK], NUM_OF_BYTES_PER_CHUNK, ptransferCount, options);
        //}
        //else if (i % 4 == 3) {
        //    status = SPI_Write(handle, &tx_buffer[3 * NUM_OF_BYTES_PER_CHUNK], NUM_OF_BYTES_PER_CHUNK, ptransferCount, options);
        //}
        
        CHUNK_NUM = i % 32;
        status = SPI_Write(handle, &tx_buffer[CHUNK_NUM * NUM_OF_BYTES_PER_CHUNK], NUM_OF_BYTES_PER_CHUNK, ptransferCount, options);
        if (i < COUNT - 1) {
            status2 = ReadFile(hComm, &RxByte, 1, &bytesWrite, NULL);
        }
        for (int j = 0; j < sizeof(rx_buffer); j += 4) {
            x = rx_buffer[j + 3] * 256 + rx_buffer[j + 2];
            y = rx_buffer[j + 1] * 256 + rx_buffer[j];
            //fprintf(fp, "y:%d x:%d ,", y, x);
            fprintf(fp, "%d,%d,", x, y);
        }
        fprintf(fp, "\n");
        printf("%d \n", RxByte);
    }
    Cleanup_libMPSSE();
    return 0;
}

int main(int argc, char** argv) {
    test(argc, argv);
}
