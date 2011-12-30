/* Shared header file among all spaces */
#include "cuShared.h"

/* Examples' headers */
#include "add_1D.h"
#include "mul_1D.h"
#include "sub_1D.h"
#include "div_1D.h"`

#include "add_2D.h"
#include "mul_2D.h"
#include "sub_2D.h"
#include "div_2D.h"







int main (int argc, char** argv)
{
	// deviceQuery::queryGPU(argc, argv); 
	
	// add_1D::run(argc, argv);
	// mul_1D::run(argc, argv);
	// sub_1D::run(argc, argv);
	// div_1D::run(argc, argv);

	// run_2D_GLOBAL_MEMORY();  

	add_2D::globalMem::run(argc, argv); 
	mul_2D::globalMem::run(argc, argv);
	sub_2D::globalMem::run(argc, argv);
	div_2D::globalMem::run(argc, argv);


	
	return 0; 
}
 