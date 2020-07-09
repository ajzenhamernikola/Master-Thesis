#include "db2_error.h"

namespace db2_sat
{

db2_sql_error::db2_sql_error(const sqlint32 &code, const std::basic_string<SQLCHAR> &reason, const std::string &file, const size_t &line)
    : code_{code}
    , reason_{reason}
    , file_{file}
    , line_{line}
{

}

std::ostream & operator<<(std::ostream & out, const db2_sat::db2_sql_error & err)
{
    out << "SQL ERROR OCCURED!" << std::endl;
    out << "In file \"" << err.file_ << " (line " << err.line_ << "):" << std::endl;
    out << "SQLCODE: " << err.code_ << std::endl;
    for(const auto c : err.reason_)
    {
        out << c;
    }
    return out;
}

}
