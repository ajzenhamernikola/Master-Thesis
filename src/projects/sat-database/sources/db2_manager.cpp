#include "db2_manager.h"

db2_sat::db2_manager::db2_manager()
{
    auto return_code = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &env_);
    CHECK_SQL_ERR(return_code, SQL_HANDLE_ENV, env_);

    return_code = SQLAllocHandle(SQL_HANDLE_DBC, env_, &connection_);
    CHECK_SQL_ERR(return_code, SQL_HANDLE_DBC, connection_);
}

db2_sat::db2_manager::~db2_manager()
{
    disconnect();

    auto return_code = SQLFreeHandle(SQL_HANDLE_DBC, connection_);
    CHECK_SQL_ERR(return_code, SQL_HANDLE_DBC, connection_);

    return_code = SQLFreeHandle(SQL_HANDLE_ENV, env_);
    CHECK_SQL_ERR(return_code, SQL_HANDLE_ENV, env_);
}

void db2_sat::db2_manager::connect(const db2_db_name &db_name, const db2_username &username, const db2_password &password)
{
    if (is_connected_)
    {
        return;
    }

    const auto return_code = SQLConnect(connection_, db_name, SQL_NTS, username, SQL_NTS, password, SQL_NTS);
    CHECK_SQL_ERR(return_code, SQL_HANDLE_DBC, connection_);

    is_connected_ = true;
}

void db2_sat::db2_manager::disconnect()
{
    if (!is_connected_)
    {
        return;
    }

    is_connected_ = false;

    auto return_code = SQLDisconnect(connection_);
    CHECK_SQL_ERR(return_code, SQL_HANDLE_DBC, connection_);

    // TODO: COMMIT
}

void db2_sat::db2_manager::check_for_sql_err(const SQLRETURN return_code, SQLSMALLINT handle_type, SQLHANDLE handle, const char * file, size_t line)
{
    SQLCHAR sqlstate[6]{};
    sqlint32 sqlcode{};
    SQLCHAR psz_error_msg[1024]{};
    SQLSMALLINT msg_length{};

    switch (return_code)
    {
    case SQL_ERROR:
    case SQL_INVALID_HANDLE:
    {
        SQLGetDiagRec(handle_type, handle, 1, sqlstate, &sqlcode, psz_error_msg, sizeof(psz_error_msg), &msg_length);

        // TODO: ROLLBACK

        const auto cropped_msg = std::basic_string<SQLCHAR>(psz_error_msg).substr(0, static_cast<size_t>(msg_length));
        std::string file_str{file};
        db2_sat::db2_sql_error err(sqlcode, cropped_msg, file_str, line);
        throw err;
    }
    default:
        break;
    }
}
