// C++ STL
#include <iostream>

// Qt API
#include <QApplication>

// Project
#include <db2_manager.h>
#include <db2_types.h>
#include "controllers/main_window.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    main_window w;
    w.show();

    auto return_code = 0;

    try {
        db2_sat::db2_manager db2;

        db2.connect(CSTR_TO_DB2_STR("sat_inst"), CSTR_TO_DB2_STR("db2admin"), CSTR_TO_DB2_STR("abcdef"));
        std::cout << "connected" << std::endl;

        return_code = a.exec();

        db2.disconnect();
        std::cout << "disconnected" << std::endl;
    } catch (const db2_sat::db2_sql_error & err) {
        std::cerr << err << std::endl;
    } catch (...) {
        std::cerr << "unknown exception" << std::endl;
    }

    return return_code;
}
