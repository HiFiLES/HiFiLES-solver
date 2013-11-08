***********************************************
**                   README                  **
***********************************************

To build the TecIO library and/or the pltview utility 
simply run the Runmake script in this directory.

If customization is needed it will most likely be done
in GLOBAL.h (to identify machine as 64 bit) and/or in
dataio4.c.  Just look for CRAY in dataio4.c and you
will find most of the critical areas.  Note that the
existing code defined by CRAY is quite old and has
not been in use for some time.

TECIO.h (which just includes TECXXX.h) and TECXXX.h 
come with the standard tecplot distribution and are 
in $TECxxxHOME/include.

tecio.inc (unix) and tecio.for (windows) are in the 
respective (unix or windows) standard tecplot 
distribution in $TECxxxHOME/util/tecio.


ReadTec()


The ReadTec() is included in the tecio library but is
not supported by Tecplot Inc.  ReadTec is used 
to read tecplot binary datafiles (all versions at or 
older than the tecplot version providing the tecio 
library).

ReadTec has the same API as TecUtilReadBinaryData.

To use ReadTec:

   1.  Read the description of TecUtilReadBinaryData in
       the Tecplot ADK Reference manual (in the doc/tecplot/
       adkrm/ directory).

   2.  Review the source code in pltview.cpp. It gives
       examples of using ReadTec to read just the header
       information from a file as well as loading all
       field data from a file.



