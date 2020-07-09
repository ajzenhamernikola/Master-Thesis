#ifndef SATDATABASE_H
#define SATDATABASE_H

// C++ STL
#include <string>

// DB2 API
#include <sqlcli1.h>

// Project
#include "db2_types.h"
#include "db2_error.h"

#define CHECK_SQL_ERR(RETCODE, HANDLE_TYPE, HANDLE) \
    check_for_sql_err(RETCODE, HANDLE_TYPE, HANDLE, __FILE__, __LINE__)

namespace db2_sat
{

class db2_manager
{
public:
    db2_manager();
    ~db2_manager();

    void connect(const db2_db_name & db_name, const db2_username & username, const db2_password & password);
    void disconnect();

private:
    void check_for_sql_err(const SQLRETURN return_code, SQLSMALLINT handle_type, SQLHANDLE handle, const char * file, size_t line);

private:
    SQLHANDLE env_{};
    SQLHANDLE connection_{};
    boolean is_connected_{false};
};

}

#endif // SATDATABASE_H
