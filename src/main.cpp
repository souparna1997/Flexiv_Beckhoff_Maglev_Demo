#include <iostream>
#include <exception>
#include "MainRedisInterface.h"
#include "controller/RobotControllerConfigParser.h"

int main() {
    try {
        SaiModel::URDF_FOLDERS["DEMO_ROBOT_FOLDER"] = DEMO_ROBOT_FOLDER;
        SaiInterfaces::MainRedisInterface interface(DEMO_CONFIG_FOLDER, "rizon_real.xml");
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
