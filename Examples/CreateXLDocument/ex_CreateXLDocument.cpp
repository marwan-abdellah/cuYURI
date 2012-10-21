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

#include "ex_CreateXLDocument.h"
#include "CUDA/cuGlobals.h"
#include "CUDA/Utilities/cuUtilities.h"
#include "CUDA/cuYURI_Interfaces.h"
#include "Globals.h"
#include "Utilities/Utils.h"
#include "MACROS/MACROS.h"
#include "MACROS/MemoryMACROS.h"
#include "Dependencies/Excel/ExcelFormat.h"

using namespace ExcelFormat;

/*! Normal font size
 */
#define	FW_NORMAL	400

/*! Bold font size
 */
#define	FW_BOLD		700

void ex_CreateXLDocument::run(int argc, char* argv[])
{

    if (argv[1] == NULL)
    {
        INFO("Usage: ex_CreateXLDocument <NUMBER_ROWS>");
        EXIT(0);
    }

    // Number of rows in the excel xlSheet
    int numRows = STI(CATS(argv[1]));

    INFO("Number of rows in ths excel sheet : " + ITS(numRows));

    // Create an excel document and set the formatting of the document
    BasicExcel xlDoc;
    XLSFormatManager fmt_mgr(xlDoc);

    // Create am excel xlSheet and get the associated BasicExcelWorksheet
    // pointer
    xlDoc.New(1);
    BasicExcelWorksheet* xlSheet = xlDoc.GetWorksheet(0);

    // Adjust header cells formatting
    ExcelFont headerFont;
    CellFormat headerFormat(fmt_mgr);
    headerFont.set_weight(FW_BOLD);
    headerFormat.set_font(headerFont);
    headerFormat.set_color1(COLOR1_PAT_SOLID);
    headerFormat.set_color2(MAKE_COLOR2(EGA_YELLOW, 0));

    // Adjust table cells formatting
    ExcelFont cellFont;
    CellFormat cellFormat(fmt_mgr);
    cellFont.set_weight(FW_NORMAL);
    cellFormat.set_font(cellFont);
    cellFormat.set_color2(MAKE_COLOR2(EGA_WHITE, 0));

    // Separator formatting
    CellFormat separatorFormat(fmt_mgr);
    separatorFormat.set_color1(COLOR1_PAT_SOLID);
    separatorFormat.set_color2(MAKE_COLOR2(EGA_RED, 0));

    // Generic cell to be used for all the excel sheet
    BasicExcelCell* xlCell;

    // Create header cells
    xlCell = xlSheet->Cell(0, 0);
    xlCell->SetFormat(headerFormat);
    xlCell->Set("Index");

    xlCell = xlSheet->Cell(0, 1);
    xlCell->SetFormat(headerFormat);
    xlCell->Set("Value");

    // Fill the table with dummy values
    for (int iRow = 1; iRow < numRows; iRow++)
    {
        // Set the index
        xlCell = xlSheet->Cell(iRow, 0);
        xlCell->SetFormat(cellFormat);
        xlCell->SetInteger(iRow);

        // Set the value
        xlCell = xlSheet->Cell(iRow, 1);
        xlCell->SetFormat(cellFormat);
        xlCell->SetInteger(iRow);
    }

    // Create a separator with RED cells
    for (int iRow = 0; iRow < numRows; iRow++)
    {
        // Set the index
        xlCell = xlSheet->Cell(iRow, 2);
        xlCell->SetFormat(separatorFormat);
    }

    // Save the XL document with the ".xlDoc" extension
    xlDoc.SaveAs("ex_CreateXLDocument.xls");

    INFO("Done");
}
