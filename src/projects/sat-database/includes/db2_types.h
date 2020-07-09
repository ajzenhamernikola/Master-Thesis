#ifndef TYPES_H
#define TYPES_H

#include <string>
#include <sql.h>

typedef SQLCHAR* DB2_STR;
typedef DB2_STR db2_db_name;
typedef DB2_STR db2_username;
typedef DB2_STR db2_password;

#define CSTR_TO_DB2_STR(c_str) reinterpret_cast<DB2_STR>(const_cast<char*>(c_str))

#endif // TYPES_H
