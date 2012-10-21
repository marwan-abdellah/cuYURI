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

#include "Logging.h"

// Log file stream
ostream_t Log::logFileStream;

/*!
 * Creates a log file to keep track of the flw of the executable.
 *
 * @param filName
 *          The name of the generated log file.
 *
 * @returns
 *          SUCCESS if the log file is generated successfully &
 *          FAIL if the file is not generated.
 */
int Log::CreateLogFile(string_t fileName)
{
    // Appending the extension string to the file name
    string_t logFileExtension = ".log";
    string_t logFileString = fileName + logFileExtension;

    // Open the log file stream
    logFileStream.open((char*)logFileString.c_str());

    // Check for a valid output file stream
    if (logFileStream != NULL)
    {
        // Logging
        LOG();
        return 0;
    }
    else
        return -1;
}


/*!
 * Closes the created log file and releases it to be opended by
 * other application.
 *
 * @returns
 *          SUCCESS if the log file is closed successfully &
 *          FAIL if the file is not closed.
 */
int Log::CloseLogFile()
{
    // Logging
    LOG();

    // Close the file stream
    logFileStream.close();

    // Check proper closure of the file
    if (logFileStream == 0)
        return 0;
    else
        return -1;
}


