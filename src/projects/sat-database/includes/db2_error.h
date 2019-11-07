#ifndef SQL_ERROR_H
#define SQL_ERROR_H

// C++ SQL
#include <iostream>
#include <string>

// Db2 API
#include <sqlcli1.h>

namespace db2_sat
{

class db2_sql_error
{
public:
    db2_sql_error(const sqlint32 & code, const std::basic_string<SQLCHAR> & reason, const std::string & file, const size_t & line);

private:
    friend std::ostream & operator<<(std::ostream & out, const db2_sql_error & err);

private:
    sqlint32 code_;
    std::basic_string<SQLCHAR> reason_;
    std::string file_;
    size_t line_;
};

std::ostream & operator<<(std::ostream & out, const db2_sat::db2_sql_error & err);
}

#endif // SQL_ERROR_H
