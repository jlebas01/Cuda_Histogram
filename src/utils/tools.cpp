//
// Created by jlebas01 on 04/04/2020.
//

#include <utils/tools.hpp>

namespace utils {

// ================================================== For image comparison
    std::ostream &operator<<(std::ostream &os, const uchar4 &c) {
        os << "[" << uint(c.x) << "," << uint(c.y) << "," << uint(c.z) << "," << uint(c.w) << "]";
        return os;
    }

    void compareImages(const std::vector<uchar4> &a, const std::vector<uchar4> &b) {
        bool error = false;
        if (a.size() != b.size()) {
            std::cout << "Size is different !" << std::endl;
            error = true;
        } else {
            for (uint i = 0; i < a.size(); ++i) {
                // Floating precision can cause small difference between host and devices
                if (std::abs(a[i].x - b[i].x) > 2 || std::abs(a[i].y - b[i].y) > 2
                    || std::abs(a[i].z - b[i].z) > 2 || std::abs(a[i].w - b[i].w) > 2) {
                    std::cout << "Error at index " << i << ": a = " << a[i] << " - b = " << b[i] << " - "
                              << std::abs(a[i].x - b[i].x) << std::endl;
                    error = true;
                    break;
                }
            }
        }
        if (error) {
            std::cout << " -> You failed, retry!" << std::endl;
        } else {
            std::cout << " -> Well done!" << std::endl;
        }
    }
}
