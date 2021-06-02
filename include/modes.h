
#ifndef MODES_H
#define MODES_H


typedef enum ReadMode_S {
    Host,
    Device
} ReadMode;

typedef enum CopyMode_s {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice
} CopyMode;


typedef enum Verbosity_s {
 SILENT,
 SUMMARIZE,
 VERBOSE,
 DEBUG_VERBOSE
} Verbosity;


typedef enum TuneMode_S {
    TUNE_NO,
    TUNE_YES
} TuneMode;




#endif
