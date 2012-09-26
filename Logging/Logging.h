/*********************************************************************
 * Copyright © 2011-2012,
 * Marwan Abdellah: <abdellah.marwan@gmail.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation.

 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.

 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 ********************************************************************/

#ifndef _LOGGING_H_
#define _LOGGING_H_

#include "Utilities/Typedefs.h"
#include "Utilities/LoggingMACROS.h"


namespace Log
{
	extern ostream_t logFileStream;

    int CreateLogFile(string_t fileName);
    void Log();
    void LogHeader(string_t Hdr);
    void LogMsg(string_t Msg);
    int CloseLogFile();
}


#endif /* _LOGGING_H_ */
