#ifndef _MESSAGING_MACROS_H_
#define _MESSAGING_MACROS_H_

/*!
 * Exits if some error has been manually caught.
 */
#define EXIT( CODE ) 															\
    COUT << STRG( __FILE__ ) << ":[" <<                             			\
    ( __LINE__ ) << "]" << ENDL << TAB <<                           			\
    STRG( __FUNCTION__ ) << ": EXITING " << CODE << ENDL;           			\
    exit( CODE );

/*!
 * Inserts a sepration line of (*) to separate between two different
 * types of messages in the output console or in the log files.
 */
#define SEP( MESSAGE )															\
    COUT << 																	\
    "********************************************************"					\
    << ENDL;

#endif // _MESSAGING_MACROS_H_
